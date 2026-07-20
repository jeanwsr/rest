use serde::{Deserialize, Serialize};

/// Parameters for ideal-gas RRHO / quasi-RRHO thermochemistry.
///
/// Triggered by a top-level `[thermo]` section. Rotational symmetry number
/// `symmetry_number` must be supplied by the user (automatic point-group
/// detection is not implemented yet). Temperature and pressure may be given
/// either as a single value or as a `[min, max, step]` scan range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermoParameters {
    /// Temperature in K (single-point primary value; scan uses `temperature_list`)
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Pressure in atm (single-point primary value)
    #[serde(default = "default_pressure")]
    pub pressure: f64,
    /// Rotational symmetry number sigma (default None, which means auto-detect from geometry)
    #[serde(default = "default_symmetry_number")]
    pub symmetry_number: f64,
    /// Electronic energy in a.u. used for U/H/G sums. If 0.0, the SCF total
    /// energy of the current calculation is used.
    #[serde(default)]
    pub electronic_energy: f64,

    // ── frequency scale factors (default 1.0 each) ──
    #[serde(default = "one")]
    pub sclzpe: f64,
    #[serde(default = "one")]
    pub sclheat: f64,
    #[serde(default = "one")]
    pub scls: f64,
    #[serde(default = "one")]
    pub sclcv: f64,

    // ── low-frequency treatment ──
    /// 0 = RRHO, 1 = Truhlar (raise), 2 = Grimme (S interp), 3 = Minenkov (S+U interp)
    #[serde(default)]
    pub ilowfreq: u32,
    /// Truhlar raising threshold (cm^-1), default 100
    #[serde(default = "default_ravib")]
    pub ravib: f64,
    /// Grimme/Minenkov interpolation threshold (cm^-1), default 100
    #[serde(default = "default_intpvib")]
    pub intpvib: f64,
    /// Treat imaginary freqs with |nu| < this (cm^-1) as real. 0 = disable.
    #[serde(default)]
    pub imagreal: f64,

    // ── scan ranges (empty => single-point using temperature/pressure) ──
    #[serde(default)]
    pub temperature_list: Vec<f64>,
    #[serde(default)]
    pub pressure_list: Vec<f64>,

    // ── concentration correction: "" or "0" = none, e.g. "1.5M" or "2.3atm" ──
    #[serde(default)]
    pub conc: String,

    /// Output path for the single-point thermochemistry report
    #[serde(default = "default_output_path")]
    pub output_path: String,
}

fn one() -> f64 { 1.0 }
fn default_temperature() -> f64 { 298.15 }
fn default_pressure() -> f64 { 1.0 }
fn default_symmetry_number() -> f64 { 0.0 }
fn default_ravib() -> f64 { 100.0 }
fn default_intpvib() -> f64 { 100.0 }
fn default_output_path() -> String { String::from("./Thermochemistry.txt") }

impl Default for ThermoParameters {
    fn default() -> Self {
        ThermoParameters {
            temperature: default_temperature(),
            pressure: default_pressure(),
            symmetry_number: default_symmetry_number(),
            electronic_energy: 0.0,
            sclzpe: one(),
            sclheat: one(),
            scls: one(),
            sclcv: one(),
            ilowfreq: 0,
            ravib: default_ravib(),
            intpvib: default_intpvib(),
            imagreal: 0.0,
            temperature_list: vec![],
            pressure_list: vec![],
            conc: String::new(),
            output_path: default_output_path(),
        }
    }
}

/// Parse a temperature/pressure value into a list of points.
/// - Number               -> single point
/// - [x]                  -> single point
/// - [min, max, step]     -> arithmetic scan range
/// - [a, b, c, ...]       -> explicit list
fn parse_tp_list(value: &serde_json::Value, default: f64) -> Vec<f64> {
    match value {
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(default)],
        serde_json::Value::Array(a) => {
            let nums: Vec<f64> = a.iter().filter_map(|x| x.as_f64()).collect();
            match nums.len() {
                0 => vec![default],
                1 => vec![nums[0]],
                3 => {
                    let (lo, hi, step) = (nums[0], nums[1], nums[2]);
                    if step <= 0.0 || hi < lo { return vec![lo]; }
                    let mut out = vec![];
                    let mut x = lo;
                    while x <= hi + step * 1.0e-6 { out.push(x); x += step; }
                    out
                }
                _ => nums,
            }
        }
        _ => vec![default],
    }
}

/// Parse a `[thermo]` section (top-level or nested under `[ctrl]`).
pub fn parse_thermo_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<ThermoParameters>> {
    let section = tmp_keys.get("thermo").or_else(|| {
        tmp_keys.get("ctrl").and_then(|c| c.get("thermo"))
    }).unwrap_or(&serde_json::Value::Null);
    match section {
        serde_json::Value::Object(o) => {
            let mut p = ThermoParameters::default();
            let get_num = |key: &str| o.get(key)
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok())));
            let get_str = |key: &str| o.get(key).and_then(|v| v.as_str()).map(|s| s.to_string());
            let get_u32 = |key: &str| o.get(key)
                .and_then(|v| v.as_u64().map(|x| x as u32)
                       .or_else(|| v.as_str().and_then(|s| s.parse::<u32>().ok())));

            // temperature / pressure: single value OR scan range
            if let Some(tv) = o.get("temperature").or_else(|| o.get("T")) {
                let list = parse_tp_list(tv, default_temperature());
                if list.len() == 1 { p.temperature = list[0]; }
                else if list.len() > 1 { p.temperature = list[0]; p.temperature_list = list; }
            }
            if let Some(pv) = o.get("pressure").or_else(|| o.get("P")) {
                let list = parse_tp_list(pv, default_pressure());
                if list.len() == 1 { p.pressure = list[0]; }
                else if list.len() > 1 { p.pressure = list[0]; p.pressure_list = list; }
            }
            if let Some(v) = get_num("symmetry_number").or_else(|| get_num("sigma")) { p.symmetry_number = v; }
            if let Some(v) = get_num("electronic_energy").or_else(|| get_num("E")) { p.electronic_energy = v; }

            // convenience single scale factor: applies to all four if present
            let single = get_num("scale_factor").or_else(|| get_num("scl"));
            if let Some(s) = single { p.sclzpe = s; p.sclheat = s; p.scls = s; p.sclcv = s; }
            if let Some(v) = get_num("sclzpe") { p.sclzpe = v; }
            if let Some(v) = get_num("sclheat") { p.sclheat = v; }
            if let Some(v) = get_num("scls") { p.scls = v; }
            if let Some(v) = get_num("sclcv") { p.sclcv = v; }

            if let Some(v) = get_u32("ilowfreq") { p.ilowfreq = v; }
            if let Some(v) = get_num("ravib") { p.ravib = v; }
            if let Some(v) = get_num("intpvib") { p.intpvib = v; }
            if let Some(v) = get_num("imagreal") { p.imagreal = v; }
            if let Some(v) = get_str("conc") { p.conc = v; }
            if let Some(v) = get_str("output_path") { p.output_path = v; }
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
