//! Ideal-gas rigid-rotor harmonic-oscillator (RRHO) and quasi-RRHO
//! thermochemistry.
//!
//! Models (Shermo-compatible, `ilowfreq`):
//!   0 - RRHO (harmonic)
//!   1 - Truhlar: raise low frequencies to `ravib` (cm^-1)
//!   2 - Grimme:  entropy interpolation between harmonic and free-rotor
//!   3 - Minenkov: interpolation of both entropy and internal energy
//!
//! Working equations follow Shermo (T. Lu, Comput. Theor. Chem. 1200, 113249,
//! 2021). All quantities are reported per mole and computed in SI units.

use std::io::Write;

use crate::constants::{self, AVOGADRO, BOLTZMANN, CLIGHT_CMS, PLANCK, R_GAS};
use crate::geom_io::get_mass_charge;
use crate::scf_io::SCF;
use crate::utilities::TimeRecords;
use tensors::MatrixFull;

// ── SI helpers ─────────────────────────────────────────────────────────
const AMU_KG: f64 = 1.0e-3 / AVOGADRO; // 1 amu in kg
const BOHR_M: f64 = constants::BOHR * 1.0e-10; // 1 Bohr in meters
const ATM_PA: f64 = 101325.0;
const AU_J: f64 = 4.3597447222071e-18; // 1 Hartree in Joule
const J_PER_MOL_TO_AU: f64 = AVOGADRO * AU_J;
const HC_JCM: f64 = PLANCK * CLIGHT_CMS; // J*cm
const C2_CMK: f64 = PLANCK * CLIGHT_CMS / BOLTZMANN; // h*c/k  [cm*K]
const BAV: f64 = 1.0e-44; // average molecular moment of inertia (kg*m^2), Grimme
const PI: f64 = std::f64::consts::PI;

// ── Public data structures ─────────────────────────────────────────────

/// Runtime input for a thermochemistry calculation.
#[derive(Debug, Clone)]
pub struct ThermoInput {
    pub temperature: f64,
    pub pressure: f64,
    pub symmetry_number: f64,
    pub electronic_energy_au: f64,
    pub sclzpe: f64,
    pub sclheat: f64,
    pub scls: f64,
    pub sclcv: f64,
    pub ilowfreq: u32, // 0=RRHO 1=Truhlar 2=Grimme 3=Minenkov
    pub ravib: f64,
    pub intpvib: f64,
    pub imagreal: f64,
    pub print_level: usize,
}

impl Default for ThermoInput {
    fn default() -> Self {
        ThermoInput {
            temperature: 298.15, pressure: 1.0, symmetry_number: 1.0,
            electronic_energy_au: 0.0, sclzpe: 1.0, sclheat: 1.0, scls: 1.0, sclcv: 1.0,
            ilowfreq: 0, ravib: 100.0, intpvib: 100.0, imagreal: 0.0, print_level: 1,
        }
    }
}

/// Molecular descriptors that are (T, P)-independent.
struct MolInfo {
    natom: usize,
    total_mass_amu: f64,
    m_kg: f64,
    principal_moments_kgm2: [f64; 3],
    principal_moments_amu_bohr2: [f64; 3],
    rot_constants_ghz: [f64; 3],
    rot_temperatures_k: [f64; 3],
    is_linear: bool,
    n_zero: usize,
    vib_freqs: Vec<f64>, // real (positive) vibrational wavenumbers (cm^-1)
    n_imag: usize,
    spin_mult: f64,
}

/// Full thermochemistry result at a single (T, P). Energies in J/mol,
/// entropies/heat capacities in J/(mol*K), partition functions dimensionless.
#[derive(Debug, Clone, Default)]
pub struct ThermoResult {
    pub natom: usize,
    pub total_mass_amu: f64,
    pub principal_moments_amu_bohr2: [f64; 3],
    pub rot_constants_ghz: [f64; 3],
    pub rot_temperatures_k: [f64; 3],
    pub is_linear: bool,
    pub symmetry_number: f64,
    pub n_real_vib_modes: usize,
    pub n_imag_vib_modes: usize,
    pub ilowfreq: u32,
    // partition functions
    pub q_trans: f64,
    pub q_rot: f64,
    pub q_vib_v0: f64,
    pub q_vib_bot: f64,
    pub q_ele: f64,
    // per-component energies (J/mol)
    pub u_trans: f64,
    pub u_rot: f64,
    pub u_vib_heat: f64, // effective U(T)-U(0) vibrational
    pub u_ele: f64,
    // per-component entropy (J/mol/K)
    pub s_trans: f64,
    pub s_rot: f64,
    pub s_vib: f64,
    pub s_ele: f64,
    // per-component heat capacity (J/mol/K)
    pub cv_trans: f64,
    pub cv_rot: f64,
    pub cv_vib: f64,
    pub cv_ele: f64,
    // totals
    pub zpe: f64,
    pub u_corr: f64,
    pub h_corr: f64,
    pub g_corr: f64,
    pub s_total: f64,
    pub cv_total: f64,
    pub cp_total: f64,
    pub conc_dg: f64, // concentration correction (added into g_corr)
    pub electronic_energy_au: f64,
    pub temperature: f64,
    pub pressure: f64,
}

// ── RRHO / free-rotor kernels ──────────────────────────────────────────

#[inline] fn vib_theta(nu_cm: f64, t: f64) -> f64 { (C2_CMK * nu_cm / t).max(1.0e-12) }

#[inline]
fn rrho_heat(nu_cm: f64, t: f64) -> f64 {
    let th = vib_theta(nu_cm, t);
    let d = (th.exp() - 1.0).max(1.0e-300);
    R_GAS * t * th / d
}
#[inline]
fn rrho_cv(nu_cm: f64, t: f64) -> f64 {
    let th = vib_theta(nu_cm, t);
    let ex = th.exp();
    let d = (ex - 1.0).max(1.0e-300);
    R_GAS * th * th * ex / (d * d)
}
#[inline]
fn rrho_s(nu_cm: f64, t: f64) -> f64 {
    let th = vib_theta(nu_cm, t);
    let ex = th.exp();
    let d = (ex - 1.0).max(1.0e-300);
    let exm = (-th).exp();
    R_GAS * (th / d - (1.0 - exm).max(1.0e-300).ln())
}
/// Grimme quasi-RRHO interpolation weight (uses the *unscaled* wavenumber).
#[inline]
fn qrrho_weight(nu_cm: f64, nu0: f64) -> f64 {
    1.0 / (1.0 + (nu0 / nu_cm.max(1.0e-9)).powi(4))
}
/// Free-rotor entropy for a 1D rotor of frequency nu_cm (unscaled).
#[inline]
fn free_rotor_s(nu_cm: f64, t: f64) -> f64 {
    let nu_hz = CLIGHT_CMS * nu_cm;
    let mu = PLANCK / (8.0 * PI * PI * nu_hz); // moment of inertia
    let mu_eff = mu * BAV / (mu + BAV);
    let arg = (8.0 * PI * PI * PI * mu_eff * BOLTZMANN * t).sqrt() / PLANCK;
    R_GAS * (0.5 + arg.max(1.0e-300).ln())
}

// ── Moment of inertia ──────────────────────────────────────────────────

fn principal_moments(
    elems: &Vec<String>,
    position: &MatrixFull<f64>,
    natm: usize,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let mass_amu = get_mass_charge(elems);
    let mut ixx = 0.0; let mut iyy = 0.0; let mut izz = 0.0;
    let mut ixy = 0.0; let mut ixz = 0.0; let mut iyz = 0.0;
    for ia in 0..natm {
        let m = mass_amu[ia].0 * AMU_KG;
        let x = position[[0, ia]] * BOHR_M;
        let y = position[[1, ia]] * BOHR_M;
        let z = position[[2, ia]] * BOHR_M;
        ixx += m * (y * y + z * z);
        iyy += m * (x * x + z * z);
        izz += m * (x * x + y * y);
        ixy -= m * x * y;
        ixz -= m * x * z;
        iyz -= m * y * z;
    }
    let data = vec![ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz];
    let mat = MatrixFull::from_vec([3, 3], data).unwrap();
    use rest_tensors::matrix::matrix_blas_lapack::_dsyev;
    let (_, eigvals, _) = _dsyev(&mat, 'N');
    let mut moments = [0.0f64; 3];
    for k in 0..3 { moments[k] = if eigvals[k] < 0.0 { 0.0 } else { eigvals[k] }; }
    let abu = AMU_KG * BOHR_M * BOHR_M;
    let m_ab = [moments[0] / abu, moments[1] / abu, moments[2] / abu];
    let factor = PLANCK / (8.0 * PI * PI);
    let ghz = [
        factor / moments[0] / 1.0e9,
        factor / moments[1] / 1.0e9,
        factor / moments[2] / 1.0e9,
    ];
    (moments, m_ab, ghz)
}

// ── Molecular info preparation (T,P-independent) ───────────────────────

fn prepare_molecular_info(
    freqs_cm1: &[f64],
    elems: &Vec<String>,
    position: &MatrixFull<f64>,
    spin_mult: f64,
    input: &ThermoInput,
) -> Result<MolInfo, String> {
    let natm = freqs_cm1.len() / 3;
    if natm == 0 || elems.len() < natm {
        return Err(format!(
            "thermo: inconsistent sizes (freqs give {} atoms, elems has {})",
            natm, elems.len()
        ));
    }
    let mass_amu = get_mass_charge(elems);
    let mut total_mass_amu = 0.0;
    for ia in 0..natm { total_mass_amu += mass_amu[ia].0; }
    let m_kg = total_mass_amu * AMU_KG;

    let (moments, m_ab, ghz) = if natm >= 2 {
        principal_moments(elems, position, natm)
    } else {
        ([0.0; 3], [0.0; 3], [0.0; 3])
    };
    let rt = [
        PLANCK * ghz[0] * 1.0e9 / BOLTZMANN,
        PLANCK * ghz[1] * 1.0e9 / BOLTZMANN,
        PLANCK * ghz[2] * 1.0e9 / BOLTZMANN,
    ];
    let is_linear = if natm < 2 { false }
        else { moments[0] < 1.0e-6 * moments[1].max(1.0e-300) };
    let n_zero = if is_linear { 5 } else { 6 };

    // Drop the n_zero lowest-magnitude (trans/rot) modes, then apply imagreal.
    let mut sorted: Vec<(f64, f64)> = freqs_cm1.iter().map(|&f| (f, f.abs())).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut vib: Vec<f64> = Vec::new();
    let mut n_imag = 0usize;
    for &(f, _) in sorted.iter().skip(n_zero) {
        if f > 0.0 {
            vib.push(f);
        } else if f < 0.0 {
            if input.imagreal > 0.0 && f.abs() < input.imagreal {
                vib.push(f.abs()); // treat small imaginary as real
            } else {
                n_imag += 1;
            }
        }
    }

    Ok(MolInfo {
        natom: natm,
        total_mass_amu,
        m_kg,
        principal_moments_kgm2: moments,
        principal_moments_amu_bohr2: m_ab,
        rot_constants_ghz: ghz,
        rot_temperatures_k: rt,
        is_linear,
        n_zero,
        vib_freqs: vib,
        n_imag,
        spin_mult,
    })
}

// ── Core: thermochemistry at one (T, P) ────────────────────────────────

fn compute_thermo_at(mi: &MolInfo, t: f64, p_pa: f64, input: &ThermoInput) -> ThermoResult {
    let sigma = input.symmetry_number.max(1.0);
    let g0 = mi.spin_mult.round().max(1.0);
    let mut res = ThermoResult::default();
    res.natom = mi.natom;
    res.temperature = t;
    res.pressure = input.pressure;
    res.symmetry_number = sigma;
    res.is_linear = mi.is_linear;
    res.ilowfreq = input.ilowfreq;
    res.total_mass_amu = mi.total_mass_amu;
    res.principal_moments_amu_bohr2 = mi.principal_moments_amu_bohr2;
    res.rot_constants_ghz = mi.rot_constants_ghz;
    res.rot_temperatures_k = mi.rot_temperatures_k;
    res.electronic_energy_au = input.electronic_energy_au;

    // electron ground-state degeneracy
    res.q_ele = g0;
    res.s_ele = R_GAS * g0.ln();

    // translation
    let a = 2.0 * PI * mi.m_kg * BOLTZMANN * t / (PLANCK * PLANCK);
    let q = a.powf(1.5) * (R_GAS * t / p_pa);
    res.q_trans = q;
    res.u_trans = 1.5 * R_GAS * t;
    res.cv_trans = 1.5 * R_GAS;
    res.s_trans = R_GAS * ((q / AVOGADRO).ln() + 2.5);

    // rotation
    if mi.natom >= 2 {
        let mom = mi.principal_moments_kgm2;
        if mi.is_linear {
            let ilin = mom[1];
            let ln_q = (8.0 * PI * PI * ilin * BOLTZMANN * t / (sigma * PLANCK * PLANCK)).ln();
            res.q_rot = ln_q.exp();
            res.u_rot = R_GAS * t;
            res.cv_rot = R_GAS;
            res.s_rot = R_GAS * (ln_q + 1.0);
        } else {
            // q_rot = (sqrt(pi)/sigma) * (8*pi^2*kT/h^2)^(3/2) * sqrt(IA*IB*IC)
            let ln_q = 0.5 * PI.ln()
                + 1.5 * (8.0 * PI * PI * BOLTZMANN * t / (PLANCK * PLANCK)).ln()
                + 0.5 * (mom[0].ln() + mom[1].ln() + mom[2].ln())
                - sigma.ln();
            res.q_rot = ln_q.exp();
            res.u_rot = 1.5 * R_GAS * t;
            res.cv_rot = 1.5 * R_GAS;
            res.s_rot = R_GAS * (ln_q + 1.5);
        }
    }

    // vibration (per mode, model-dependent)
    let mut zpe = 0.0;
    let mut u_corr_vib = 0.0;
    let mut cv = 0.0;
    let mut s = 0.0;
    let mut ln_qv0 = 0.0;
    let mut ln_qvbot = 0.0;
    for &f in &mi.vib_freqs {
        let fz = f * input.sclzpe;
        let fh = f * input.sclheat;
        let fs = f * input.scls;
        let fc = f * input.sclcv;
        let zpe_m = 0.5 * AVOGADRO * HC_JCM * fz;
        zpe += zpe_m;

        // heat capacity: RRHO always (Truhlar uses raised frequency)
        let cv_m = if input.ilowfreq == 1 {
            rrho_cv(f.max(input.ravib) * input.sclcv, t)
        } else {
            rrho_cv(fc, t)
        };
        cv += cv_m;

        // harmonic q_vib for reporting (uses sclheat freq)
        {
            let th = vib_theta(fh.max(1.0e-9), t);
            let exm = (-th).exp();
            let one_m = (1.0 - exm).max(1.0e-300);
            ln_qv0 += -one_m.ln();
            ln_qvbot += -th * 0.5 - one_m.ln();
        }

        match input.ilowfreq {
            0 => { // RRHO
                u_corr_vib += zpe_m + rrho_heat(fh, t);
                s += rrho_s(fs, t);
            }
            1 => { // Truhlar: ZPE unraised; U0->T, S use raised freq
                let frh = f.max(input.ravib) * input.sclheat;
                let frs = f.max(input.ravib) * input.scls;
                u_corr_vib += zpe_m + rrho_heat(frh, t);
                s += rrho_s(frs, t);
            }
            2 => { // Grimme: entropy interpolation only
                let w = qrrho_weight(f, input.intpvib);
                u_corr_vib += zpe_m + rrho_heat(fh, t);
                s += w * rrho_s(fs, t) + (1.0 - w) * free_rotor_s(f, t);
            }
            3 => { // Minenkov: interpolate S and U
                let w = qrrho_weight(f, input.intpvib);
                let u_rrho_full = zpe_m + rrho_heat(fh, t);
                u_corr_vib += w * u_rrho_full + (1.0 - w) * 0.5 * R_GAS * t;
                s += w * rrho_s(fs, t) + (1.0 - w) * free_rotor_s(f, t);
            }
            _ => {}
        }
    }
    res.n_real_vib_modes = mi.vib_freqs.len();
    res.n_imag_vib_modes = mi.n_imag;
    res.zpe = zpe;
    res.u_vib_heat = u_corr_vib - zpe;
    res.cv_vib = cv;
    res.s_vib = s;
    res.q_vib_v0 = ln_qv0.exp();
    res.q_vib_bot = ln_qvbot.exp();

    // totals
    res.u_corr = u_corr_vib + res.u_trans + res.u_rot + res.u_ele;
    res.h_corr = res.u_corr + R_GAS * t;
    res.s_total = res.s_trans + res.s_rot + res.s_vib + res.s_ele;
    res.cv_total = res.cv_trans + res.cv_rot + res.cv_vib + res.cv_ele;
    res.cp_total = res.cv_total + R_GAS;
    res.g_corr = res.h_corr - t * res.s_total;
    res
}

/// Compute thermochemistry at a single (T, P). Convenience wrapper around
/// `prepare_molecular_info` + `compute_thermo_at`.
pub fn compute_thermochemistry(
    freqs_cm1: &[f64],
    elems: &Vec<String>,
    position: &MatrixFull<f64>,
    spin_multiplicity: f64,
    input: &ThermoInput,
) -> Result<ThermoResult, String> {
    let mi = prepare_molecular_info(freqs_cm1, elems, position, spin_multiplicity, input)?;
    Ok(compute_thermo_at(&mi, input.temperature, input.pressure * ATM_PA, input))
}

// ── Concentration correction ───────────────────────────────────────────

/// Parse a concentration spec ("1.5M", "2.3atm", or a bare number in M) and
/// return cB in mol/m^3. Returns None if no correction requested.
fn parse_conc(spec: &str) -> Option<(f64, String)> {
    let s = spec.trim();
    if s.is_empty() || s == "0" { return None; }
    if let Some(num) = s.strip_suffix("M").or_else(|| s.strip_suffix("m")) {
        let mol: f64 = num.trim().parse().ok()?;
        Some((mol * 1000.0, format!("{} M", mol)))
    } else if let Some(num) = s.strip_suffix("atm") {
        let atm: f64 = num.trim().parse().ok()?;
        Some((atm, format!("{} atm", atm))) // returned as atm; converted later
    } else {
        let mol: f64 = s.parse().ok()?;
        Some((mol * 1000.0, format!("{} M", mol)))
    }
}

/// Apply concentration correction ΔG = RT ln(cB/cA) (ideal gas, cA = P/RT).
/// `cb_atm_or_m3`: for "atm" specs the first tuple element holds atm; for "M"
/// it holds mol/m^3. We disambiguate via the description string.
fn conc_dg(spec: &str, t: f64, p_pa: f64) -> Option<(f64, String)> {
    let (val, desc) = parse_conc(spec)?;
    let c_a = p_pa / (R_GAS * t); // mol/m^3
    let c_b = if desc.ends_with("atm") {
        let atm = val;
        atm * ATM_PA / (R_GAS * t)
    } else {
        val // already mol/m^3
    };
    let dg = R_GAS * t * (c_b / c_a).ln();
    Some((dg, desc))
}

// ── Unit conversions & printing ────────────────────────────────────────

fn cal(j: f64) -> f64 { j / 4.184 }
fn kcal(j: f64) -> f64 { j / 4184.0 }
fn au(j: f64) -> f64 { j / J_PER_MOL_TO_AU }

fn model_name(ilowfreq: u32) -> &'static str {
    match ilowfreq {
        0 => "RRHO",
        1 => "Truhlar (raise low freq)",
        2 => "Grimme quasi-RRHO (S)",
        3 => "Minenkov quasi-RRHO (S+U)",
        _ => "unknown",
    }
}

pub fn print_thermo(res: &ThermoResult, pl: usize) {
    let line_u = |w: &mut String, label: &str, v: f64| {
        w.push_str(&format!("  {:<15} {:>12.3} {:>12.3} {:>14.6}\n",
            label, v / 1000.0, kcal(v), au(v)));
    };
    let line_s = |w: &mut String, label: &str, v: f64| {
        w.push_str(&format!("  {:<15} {:>12.3} {:>12.3}\n", label, v, cal(v)));
    };
    let mut out = String::new();
    out.push_str(&format!("\n=== Thermochemistry [{}]  T={:.3} K  P={:.4} atm ===\n",
        model_name(res.ilowfreq), res.temperature, res.pressure));
    out.push_str(&format!("  Atoms: {}   Molecular mass: {:.6} amu\n",
        res.natom, res.total_mass_amu));
    if res.natom >= 2 {
        let m = res.principal_moments_amu_bohr2;
        let g = res.rot_constants_ghz;
        let rt = res.rot_temperatures_k;
        out.push_str(&format!("  Principal moments of inertia (amu*Bohr^2): {:>12.6} {:>12.6} {:>12.6}\n", m[0], m[1], m[2]));
        out.push_str(&format!("  Rotational constants (GHz):               {:>12.6} {:>12.6} {:>12.6}\n", g[0], g[1], g[2]));
        out.push_str(&format!("  Rotational temperatures (K):              {:>12.6} {:>12.6} {:>12.6}\n", rt[0], rt[1], rt[2]));
        out.push_str(&format!("  Linear molecule: {}   sigma = {}\n", res.is_linear, res.symmetry_number));
    }
    out.push_str(&format!("  Vibrational modes used: {} real ({} imaginary ignored)\n",
        res.n_real_vib_modes, res.n_imag_vib_modes));
    if pl > 0 {
        out.push_str(&format!(
            "\n  Partition functions:  q_trans={:.6e}  q_rot={:.6e}  q_vib(V=0)={:.6e}  q_vib(bot)={:.6e}  q_ele={:.6}\n",
            res.q_trans, res.q_rot, res.q_vib_v0, res.q_vib_bot, res.q_ele));
    }
    out.push_str(&format!("\n  {:<15} {:>12} {:>12} {:>14}\n", "", "kJ/mol", "kcal/mol", "a.u."));
    line_u(&mut out, "ZPE", res.zpe);
    line_u(&mut out, "U_trans", res.u_trans);
    line_u(&mut out, "U_rot", res.u_rot);
    line_u(&mut out, "U_vib(T)-U(0)", res.u_vib_heat);
    line_u(&mut out, "U_corr(T)", res.u_corr);
    line_u(&mut out, "H_corr(T)", res.h_corr);
    line_u(&mut out, "G_corr(T)", res.g_corr);
    if res.conc_dg.abs() > 0.0 {
        line_u(&mut out, "  incl. dG_conc", res.g_corr);
    }
    out.push_str(&format!("\n  {:<15} {:>12} {:>12}\n", "", "J/mol/K", "cal/mol/K"));
    line_s(&mut out, "S_trans", res.s_trans);
    line_s(&mut out, "S_rot", res.s_rot);
    line_s(&mut out, "S_vib", res.s_vib);
    line_s(&mut out, "S_ele", res.s_ele);
    line_s(&mut out, "S_total", res.s_total);
    line_s(&mut out, "CV", res.cv_total);
    line_s(&mut out, "CP", res.cp_total);

    let e_au = res.electronic_energy_au;
    out.push_str(&format!("\n  Electronic energy: {:.6} a.u.\n", e_au));
    out.push_str(&format!("  Sum E + ZPE    (= U/H/G at 0 K)  : {:.6} a.u.\n", e_au + au(res.zpe)));
    out.push_str(&format!("  Sum E + U_corr (= U at {:.0} K)    : {:.6} a.u.\n", res.temperature, e_au + au(res.u_corr)));
    out.push_str(&format!("  Sum E + H_corr (= H at {:.0} K)    : {:.6} a.u.\n", res.temperature, e_au + au(res.h_corr)));
    out.push_str(&format!("  Sum E + G_corr (= G at {:.0} K)    : {:.6} a.u.\n", res.temperature, e_au + au(res.g_corr)));
    println!("{}", out);
}

fn write_single_report(res: &ThermoResult, path: &str, e_au: f64) {
    let mut r = String::new();
    r.push_str(&format!("# REST thermochemistry report [{}]\n", model_name(res.ilowfreq)));
    r.push_str(&format!("# T = {:.4} K, P = {:.4} atm, sigma = {}\n",
        res.temperature, res.pressure, res.symmetry_number));
    r.push_str(&format!("# natom = {}, mass = {:.6} amu, linear = {}\n",
        res.natom, res.total_mass_amu, res.is_linear));
    r.push_str(&format!("# vib modes: {} real, {} imaginary ignored\n",
        res.n_real_vib_modes, res.n_imag_vib_modes));
    let row = |k: &str, j: f64| format!("# {:<14} {:14.6} a.u.  {:12.3} kJ/mol  {:12.3} kcal/mol\n",
        k, au(j), j / 1000.0, kcal(j));
    r.push_str(&row("ZPE", res.zpe));
    r.push_str(&row("U_corr(T)", res.u_corr));
    r.push_str(&row("H_corr(T)", res.h_corr));
    r.push_str(&row("G_corr(T)", res.g_corr));
    if res.conc_dg.abs() > 0.0 {
        r.push_str(&format!("#   (includes dG_conc = {:.3} kJ/mol)\n", res.conc_dg / 1000.0));
    }
    r.push_str(&format!("# {:<14} {:12.4} J/mol/K  {:12.4} cal/mol/K\n", "S_total", res.s_total, cal(res.s_total)));
    r.push_str(&format!("# {:<14} {:12.4} J/mol/K  {:12.4} cal/mol/K\n", "CV", res.cv_total, cal(res.cv_total)));
    r.push_str(&format!("# {:<14} {:12.4} J/mol/K  {:12.4} cal/mol/K\n", "CP", res.cp_total, cal(res.cp_total)));
    r.push_str(&format!("# {:<14} {:.6} a.u.\n", "E_elec", e_au));
    r.push_str(&format!("# {:<14} {:.6} a.u.\n", "E + ZPE", e_au + au(res.zpe)));
    r.push_str(&format!("# {:<14} {:.6} a.u.\n", "E + U_corr", e_au + au(res.u_corr)));
    r.push_str(&format!("# {:<14} {:.6} a.u.\n", "E + H_corr", e_au + au(res.h_corr)));
    r.push_str(&format!("# {:<14} {:.6} a.u.\n", "E + G_corr", e_au + au(res.g_corr)));
    match std::fs::File::create(path) {
        Ok(mut f) => { let _ = f.write_all(r.as_bytes()); }
        Err(e) => eprintln!("  WARNING: failed to write {}: {}", path, e),
    }
}

// ── Scanning ───────────────────────────────────────────────────────────

fn run_scan(
    mi: &MolInfo,
    ts: &[f64],
    ps: &[f64],
    input: &ThermoInput,
    scq_path: &str,
    uhg_path: &str,
) {
    let mut scq = String::new();
    let mut uhg = String::new();
    scq.push_str("# unit of S, CV, CP: cal/mol/K; q(V=0)/NA, q(bot)/NA dimensionless\n");
    scq.push_str(&format!("{:<10} {:<10} {:>12} {:>12} {:>12} {:>16} {:>16}\n",
        "T(K)", "P(atm)", "S", "CV", "CP", "q(V=0)/NA", "q(bot)/NA"));
    uhg.push_str("# Ucorr/Hcorr/Gcorr in kcal/mol; U/H/G in a.u.\n");
    uhg.push_str(&format!("{:<10} {:<10} {:>12} {:>12} {:>12} {:>16} {:>16} {:>16}\n",
        "T(K)", "P(atm)", "Ucorr", "Hcorr", "Gcorr", "U", "H", "G"));
    let e_au = input.electronic_energy_au;
    for &t in ts {
        for &p_atm in ps {
            let p_pa = p_atm * ATM_PA;
            let mut inp = input.clone();
            inp.temperature = t;
            inp.pressure = p_atm;
            let r = compute_thermo_at(mi, t, p_pa, &inp);
            scq.push_str(&format!("{:<10.3} {:<10.4} {:12.4} {:12.4} {:12.4} {:16.8e} {:16.8e}\n",
                t, p_atm, cal(r.s_total), cal(r.cv_total), cal(r.cp_total),
                r.q_vib_v0 / AVOGADRO, r.q_vib_bot / AVOGADRO));
            uhg.push_str(&format!("{:<10.3} {:<10.4} {:12.4} {:12.4} {:12.4} {:16.8} {:16.8} {:16.8}\n",
                t, p_atm, kcal(r.u_corr), kcal(r.h_corr), kcal(r.g_corr),
                e_au + au(r.u_corr), e_au + au(r.h_corr), e_au + au(r.g_corr)));
        }
    }
    match std::fs::File::create(scq_path) {
        Ok(mut f) => { let _ = f.write_all(scq.as_bytes()); }
        Err(e) => eprintln!("  WARNING: failed to write {}: {}", scq_path, e),
    }
    match std::fs::File::create(uhg_path) {
        Ok(mut f) => { let _ = f.write_all(uhg.as_bytes()); }
        Err(e) => eprintln!("  WARNING: failed to write {}: {}", uhg_path, e),
    }
}

// ── Entry point from the Hessian pipeline ──────────────────────────────

/// Run a thermochemistry calculation from SCF data + harmonic frequencies.
/// Dispatches to single-point (with optional concentration correction) or
/// temperature/pressure scanning depending on the input parameters.
pub fn run_thermochemistry(
    scf: &SCF,
    freqs_cm1: &[f64],
    params: &crate::ctrl_io::thermo_parameters::ThermoParameters,
    time_mark: &mut TimeRecords,
) {
    let pl = scf.mol.ctrl.print_level;
    let mol = &scf.mol;
    let e_au = if params.electronic_energy != 0.0 { params.electronic_energy } else { scf.scf_energy };
    let input = ThermoInput {
        temperature: params.temperature,
        pressure: params.pressure,
        symmetry_number: params.symmetry_number,
        electronic_energy_au: e_au,
        sclzpe: params.sclzpe,
        sclheat: params.sclheat,
        scls: params.scls,
        sclcv: params.sclcv,
        ilowfreq: params.ilowfreq,
        ravib: params.ravib,
        intpvib: params.intpvib,
        imagreal: params.imagreal,
        print_level: pl,
    };

    time_mark.new_item("Thermochemistry", "RRHO thermochemistry");
    time_mark.count_start("Thermochemistry");
    let t0 = std::time::Instant::now();

    let mi = match prepare_molecular_info(
        freqs_cm1, &mol.geom.elem, &mol.geom.position, mol.ctrl.spin, &input,
    ) {
        Ok(m) => m,
        Err(e) => { eprintln!("Error in thermochemistry: {}", e); return; }
    };

    let ts: Vec<f64> = if !params.temperature_list.is_empty() { params.temperature_list.clone() }
                       else { vec![params.temperature] };
    let ps: Vec<f64> = if !params.pressure_list.is_empty() { params.pressure_list.clone() }
                      else { vec![params.pressure] };
    let scan = ts.len() > 1 || ps.len() > 1;

    if pl > 0 {
        println!("\n=== Thermochemistry Calculation [{}] ===", model_name(params.ilowfreq));
    }

    if scan {
        // scan mode: write scan_SCq.txt and scan_UHG.txt (no concentration term)
        let scq_path = "scan_SCq.txt";
        let uhg_path = "scan_UHG.txt";
        run_scan(&mi, &ts, &ps, &input, scq_path, uhg_path);
        if pl > 0 {
            println!("  Scan over {} T point(s) x {} P point(s) = {} evaluations",
                ts.len(), ps.len(), ts.len() * ps.len());
            println!("  S/CV/CP + q written to {}", scq_path);
            println!("  U/H/G corrections written to {}", uhg_path);
        }
    } else {
        // single-point evaluation
        let t = ts[0];
        let p_pa = ps[0] * ATM_PA;
        let mut res = compute_thermo_at(&mi, t, p_pa, &input);
        // concentration correction (single-point only)
        if !params.conc.is_empty() && params.conc != "0" {
            if let Some((dg, desc)) = conc_dg(&params.conc, t, p_pa) {
                if pl > 0 {
                    println!("  Concentration correction to {}: dG = {:.3} kJ/mol", desc, dg / 1000.0);
                }
                res.conc_dg = dg;
                res.g_corr += dg;
            }
        }
        print_thermo(&res, pl);
        write_single_report(&res, &params.output_path, e_au);
    }

    time_mark.count("Thermochemistry");
    if pl > 0 {
        println!("  Thermochemistry elapsed: {:.3} s", t0.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos_matrix(natm: usize, flat: &[f64]) -> MatrixFull<f64> {
        let mut m = MatrixFull::new([3, natm], 0.0);
        for ia in 0..natm {
            for c in 0..3 { m[[c, ia]] = flat[ia * 3 + c]; }
        }
        m
    }

    /// Cross-check against Shermo's published H2CO reference (manual Sec. 3.1).
    #[test]
    fn h2co_matches_shermo_reference() {
        let freqs_full = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 6 near-zero trans/rot modes
            1210.2, 1273.5, 1544.3, 1819.4, 2887.7, 2945.7,
        ];
        let elems: Vec<String> = vec!["C".into(), "O".into(), "H".into(), "H".into()];
        let flat = [0.0,0.0,0.0, 0.0,0.0,2.276, 0.0,1.768,-1.023, 0.0,-1.768,-1.023];
        let position = pos_matrix(4, &flat);

        // ZPE with sclzpe=0.9806
        let inp = ThermoInput { temperature: 350.0, pressure: 1.0, symmetry_number: 2.0,
            sclzpe: 0.9806, ..Default::default() };
        let r = compute_thermochemistry(&freqs_full, &elems, &position, 1.0, &inp).unwrap();
        assert!((r.zpe / 1000.0 - 68.51).abs() < 0.1, "ZPE {}", r.zpe / 1000.0);

        // Vibration/trans/rot with scale factors = 1.0 (RRHO)
        let inp2 = ThermoInput { sclzpe: 1.0, ..inp.clone() };
        let r2 = compute_thermochemistry(&freqs_full, &elems, &position, 1.0, &inp2).unwrap();
        assert!((r2.u_vib_heat / 1000.0 - 0.227).abs() < 0.005, "U_vib_heat {}", r2.u_vib_heat);
        assert!((r2.cv_vib - 3.534).abs() < 0.01, "CV_vib {}", r2.cv_vib);
        assert!((r2.s_vib - 0.771).abs() < 0.005, "S_vib {}", r2.s_vib);
        assert!((r2.u_trans / 1000.0 - 4.365).abs() < 0.001, "U_trans {}", r2.u_trans);
        assert!((r2.cv_trans - 12.472).abs() < 0.001, "CV_trans {}", r2.cv_trans);
        assert!((r2.s_trans - 154.5).abs() < 1.0, "S_trans {}", r2.s_trans);
        assert!(r2.q_trans > 1.0e29, "q_trans {}", r2.q_trans);
        assert!(!r2.is_linear, "H2CO should be nonlinear");
        assert_eq!(r2.n_real_vib_modes, 6);
        assert!((r2.cv_total - 28.478).abs() < 0.02, "CV_total {}", r2.cv_total);
        assert!((r2.cp_total - 36.792).abs() < 0.02, "CP_total {}", r2.cp_total);

        // quasi-RRHO must not blow up for low frequencies: feed a 20 cm^-1 mode
        let low = vec![0.0; 6]; // linear 5 + extra; use nonlinear n_zero=6
        let mut fl = low.clone();
        fl.extend_from_slice(&[20.0, 1500.0]);
        let g = ThermoInput { ilowfreq: 2, scls: 1.0, sclheat: 1.0, sclzpe: 1.0, sclcv: 1.0,
            temperature: 298.15, pressure: 1.0, symmetry_number: 1.0, ..Default::default() };
        let rg = compute_thermochemistry(&fl, &elems, &position, 1.0, &g).unwrap();
        // Grimme S for 20 cm^-1 must be finite and not the divergent RRHO value
        assert!(rg.s_vib.is_finite() && rg.s_vib > 0.0 && rg.s_vib < 50.0, "Grimme S_vib {}", rg.s_vib);
        // Minenkov U must be finite
        let gm = ThermoInput { ilowfreq: 3, ..g.clone() };
        let rm = compute_thermochemistry(&fl, &elems, &position, 1.0, &gm).unwrap();
        assert!(rm.u_corr.is_finite(), "Minenkov U_corr not finite");
    }
}
