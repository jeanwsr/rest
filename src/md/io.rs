use std::fs::File;
use std::io::Write;

use super::integrator::{AU_FORCE2_EV_PER_ANG, AU_TIME_TO_FS};

const AU_VEL_TO_ANG_PER_FS: f64 = crate::constants::BOHR / AU_TIME_TO_FS;

pub fn write_md_log_line<W: Write>(
    file: &mut W,
    step: usize,
    time_fs: f64,
    epot_ev: f64,
    ekin_ev: f64,
    etot_ev: f64,
    temp_k: f64,
    dipole_debye: Option<[f64; 3]>,
) -> std::io::Result<()> {
    let dip = match dipole_debye {
        Some(d) => format!("  dipole={:.6},{:.6},{:.6}", d[0], d[1], d[2]),
        None => String::new(),
    };
    writeln!(
        file,
        "{:6}  {:10.3} fs  Epot={:15.6} eV  Ekin={:15.6} eV  Etot={:15.6} eV  T={:10.2} K{}",
        step, time_fs, epot_ev, ekin_ev, etot_ev, temp_k, dip
    )
}

pub fn write_dipole_line<W: Write>(file: &mut W, dipole_debye: [f64; 3]) -> std::io::Result<()> {
    writeln!(file, "{:.12e} {:.12e} {:.12e}", dipole_debye[0], dipole_debye[1], dipole_debye[2])
}

#[allow(clippy::too_many_arguments)]
pub fn write_traj_frame<W: Write>(
    file: &mut W,
    symbols: &[String],
    pos_bohr: &[f64],
    vel_au: &[f64],
    force_au: &[f64],
    mass_amu: &[f64],
    step: usize,
    energy_ev: f64,
    dipole_debye: Option<[f64; 3]>,
    pbc: bool,
) -> std::io::Result<()> {
    let dip = match dipole_debye {
        Some(d) => format!(" dipole=\"{:.8} {:.8} {:.8}\"", d[0], d[1], d[2]),
        None => String::new(),
    };
    let pbc_str = if pbc { "T T T" } else { "F F F" };
    writeln!(file, "{}", symbols.len())?;
    writeln!(
        file,
        "Properties=species:S:1:pos:R:3:momenta:R:3:forces:R:3 momenta_units=\"amu*A/fs\" step={} energy={:.12}{} pbc=\"{}\"",
        step, energy_ev, dip, pbc_str
    )?;
    for i in 0..symbols.len() {
        let (mut x, mut y, mut z) = (pos_bohr[3 * i], pos_bohr[3 * i + 1], pos_bohr[3 * i + 2]);

        x *= crate::constants::BOHR;
        y *= crate::constants::BOHR;
        z *= crate::constants::BOHR;

        let vf = AU_VEL_TO_ANG_PER_FS;
        let (px, py, pz) = (
            mass_amu[i] * vel_au[3 * i] * vf,
            mass_amu[i] * vel_au[3 * i + 1] * vf,
            mass_amu[i] * vel_au[3 * i + 2] * vf,
        );
        let ff = AU_FORCE2_EV_PER_ANG;
        let (fx, fy, fz) = (force_au[3 * i] * ff, force_au[3 * i + 1] * ff, force_au[3 * i + 2] * ff);
        writeln!(
            file,
            "{:<2}{:15.8}{:15.8}{:15.8}{:15.8}{:15.8}{:15.8}{:15.8}{:15.8}{:15.8}",
            symbols[i], x, y, z, px, py, pz, fx, fy, fz
        )?;
    }
    Ok(())
}

pub fn write_energy_force_header<W: Write>(file: &mut W) -> std::io::Result<()> {
    writeln!(file, "# step E_QM_eV E_MM_eV E_tot_eV max_F_QM max_F_MM")
}

pub fn write_energy_force_line<W: Write>(
    file: &mut W,
    step: usize,
    e_qm_ev: f64,
    e_mm_ev: f64,
    e_tot_ev: f64,
    max_f_qm_ev: f64,
    max_f_mm_ev: f64,
) -> std::io::Result<()> {
    writeln!(
        file,
        "{} {:.9} {:.9} {:.9} {:.6} {:.6}",
        step, e_qm_ev, e_mm_ev, e_tot_ev, max_f_qm_ev, max_f_mm_ev
    )
}

pub fn write_umbrella_header<W: Write>(file: &mut W, cv_cols: &[&str]) -> std::io::Result<()> {
    writeln!(
        file,
        "step,time_fs,{},bias_eV,total_eV,temperature_K",
        cv_cols.join(",")
    )
}

pub fn write_umbrella_line<W: Write>(
    file: &mut W,
    step: usize,
    time_fs: f64,
    cv_values: &[f64],
    bias_ev: f64,
    total_ev: f64,
    temp_k: f64,
) -> std::io::Result<()> {
    let cv: Vec<String> = cv_values.iter().map(|v| format!("{:.4}", v)).collect();
    writeln!(
        file,
        "{},{:.3},{},{:.8},{:.8},{:.2}",
        step, time_fs, cv.join(","), bias_ev, total_ev, temp_k
    )
}

pub fn write_restart(
    path: &std::path::Path,
    step: usize,
    dt_fs: f64,
    symbols: &[String],
    pos_bohr: &[f64],
    vel_au: &[f64],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "# REST md restart: step dt_fs natoms")?;
    writeln!(f, "{} {} {}", step, dt_fs, symbols.len())?;
    writeln!(f, "# Sym x_A y_A z_A vx_A/fs vy_A/fs vz_A/fs")?;
    let vf = AU_VEL_TO_ANG_PER_FS;
    for i in 0..symbols.len() {
        writeln!(
            f,
            "{:<2}{:20.12}{:20.12}{:20.12}{:20.12}{:20.12}{:20.12}",
            symbols[i],
            pos_bohr[3 * i] * crate::constants::BOHR,
            pos_bohr[3 * i + 1] * crate::constants::BOHR,
            pos_bohr[3 * i + 2] * crate::constants::BOHR,
            vel_au[3 * i] * vf,
            vel_au[3 * i + 1] * vf,
            vel_au[3 * i + 2] * vf
        )?;
    }
    Ok(())
}

pub fn read_restart(path: &str, symbols: &[String]) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read restart file '{}': {}", path, e))?;
    let mut pos: Vec<f64> = Vec::with_capacity(3 * symbols.len());
    let mut vel: Vec<f64> = Vec::with_capacity(3 * symbols.len());
    let mut n_rows = 0usize;
    let vel_au_per_angfs = AU_TIME_TO_FS / crate::constants::BOHR;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();

        if fields.first().map_or(true, |f| f.parse::<f64>().is_ok()) {
            continue;
        }
        let nums: Vec<f64> = fields[1..]
            .iter()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        if nums.len() < 3 {
            return Err(anyhow::anyhow!(
                "restart file '{}' row {} has {} numeric columns; expected \
                 'Sym x y z [vx vy vz]' (Angstrom, Angstrom/fs)",
                path,
                n_rows + 1,
                nums.len()
            ));
        }
        for c in 0..3 {
            pos.push(nums[c] / crate::constants::BOHR);
        }
        if nums.len() >= 6 {
            for c in 3..6 {
                vel.push(nums[c] * vel_au_per_angfs);
            }
        } else {
            vel.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        n_rows += 1;
    }
    if n_rows != symbols.len() {
        return Err(anyhow::anyhow!(
            "restart file '{}' has {} atom rows but the system has {} atoms",
            path,
            n_rows,
            symbols.len()
        ));
    }
    Ok((pos, vel))
}

pub fn read_velocities_file(path: &str, symbols: &[String]) -> anyhow::Result<Vec<f64>> {
    let content = std::fs::read_to_string(path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            continue;
        }

        if fields[0].parse::<f64>().is_ok() {
            continue;
        }
        let nums: Vec<f64> = fields[1..]
            .iter()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        if nums.len() >= 3 {
            rows.push(nums);
        }
    }
    if rows.len() != symbols.len() {
        return Err(anyhow::anyhow!(
            "velocities_file has {} atom rows but the system has {} atoms",
            rows.len(),
            symbols.len()
        ));
    }

    let masses_amu: Vec<f64> =
        crate::geom_io::get_mass_charge(&symbols.to_vec()).iter().map(|(m, _)| *m).collect();
    let vel_au_per_angfs = AU_TIME_TO_FS / crate::constants::BOHR;
    let mut vel = Vec::with_capacity(3 * symbols.len());
    for (i, nums) in rows.iter().enumerate() {

        let v_angfs: [f64; 3] = match nums.len() {
            3 => [nums[0], nums[1], nums[2]],
            6 => [nums[3], nums[4], nums[5]],
            9 => [nums[3] / masses_amu[i], nums[4] / masses_amu[i], nums[5] / masses_amu[i]],
            _ => {
                return Err(anyhow::anyhow!(
                    "velocities_file row {} has {} numeric columns; expected 3 (velocities), \
                     6 (positions + velocities) or 9 (extxyz with momenta)",
                    i + 1,
                    nums.len()
                ))
            }
        };
        for c in 0..3 {
            vel.push(v_angfs[c] * vel_au_per_angfs);
        }
    }
    Ok(vel)
}

// ---------------------------------------------------------------------------
// PDB trajectory output (template driven)
// ---------------------------------------------------------------------------

/// Atom metadata read from a reference PDB.
///
/// Parsing follows the PDB fixed-column specification (records ATOM/HETATM):
///   cols  1-6  record name
///   cols 13-16 atom name   (kept verbatim so output reproduces the input)
///   col  17    altLoc      (only blank / 'A' are kept)
///   cols 18-20 residue name
///   col  22    chain id
///   cols 23-26 residue sequence number (kept verbatim; hex/hybrid-36 safe)
///   cols 31-54 x, y, z
///   cols 77-78 element symbol
/// Only the first MODEL of a multi-model file is used.  Files without the
/// element column are handled by an inference chain that follows the usual
/// PDB conventions (ion residue names / known two-letter symbols / column
/// alignment / first letter).
pub struct PdbTemplate {
    pub names_raw: Vec<String>,
    pub resnames: Vec<String>,
    pub resids: Vec<String>,
    pub chains: Vec<String>,
    pub elements: Vec<String>,
    pub elems_raw: Vec<String>,
    pub solvent: Vec<bool>,
    pub positions: Vec<[f64; 3]>,
}

fn field_str(line: &[u8], a: usize, b: usize) -> String {
    if a >= line.len() {
        return String::new();
    }
    let end = b.min(line.len());
    String::from_utf8_lossy(&line[a..end]).to_string()
}

pub(crate) const TWO_LETTER_ELEMENTS: &[&str] = &[
    "NA", "MG", "CL", "BR", "ZN", "FE", "MN", "CO", "NI", "CU", "CD", "LI", "RB", "CS", "SR",
    "BA", "AL", "SI", "SE", "CA", "AG", "AU", "HG", "PT", "PD", "SN", "PB", "TI", "CR", "MO",
];

/// Residue names denoting ions/metals: their atom names carry a two-letter
/// element symbol (NA, CL, MG, CA...).  In non-ion residues the same
/// spellings are organic atom labels (protein CA = alpha carbon,
/// PB = P-beta, HG = H-gamma) whose element is only the first letter.
/// Covers the PDB / CHARMM / GROMACS / AMBER spellings of common ions.
pub(crate) const ION_RESNAMES: &[&str] = &[
    "NA", "CL", "MG", "CA", "K", "ZN", "FE", "MN", "CO", "NI", "CU", "CD", "HG", "PB",
    "SOD", "CLA", "CAL", "POT", "MG2", "ZN2", "FE2", "FE3", "CD2", "HG2", "PB2",
    "CU1", "CU2", "CUA", "MN2", "MN3", "AL", "SR", "BA", "CS", "RB", "LI", "BR",
    "I", "IOD", "SM", "GD", "YB", "MO", "RU", "RH", "PD", "AG", "OS", "IR",
    "PT", "AU", "NB", "ZR", "BE", "GA", "GE", "EU", "TB", "DY", "ER", "TM", "LU",
    "NA+", "CL-", "MG2+", "K+", "CA2+", "ZN2+", "FE2+", "FE3+",
];

pub(crate) const WATER_RESNAMES: &[&str] =
    &["SOL", "HOH", "WAT", "TIP3", "TIP", "TIP3P", "TIP4P", "SPC", "DOD", "H2O"];

/// Atom-name spellings that are unambiguously elemental: no organic atom
/// label uses them, so a match always denotes the element.  Covers heme iron
/// (HEM/FE), selenomethionine (MSE/SE), ligand halogens (CL, BR) and the
/// common transition metals.  The four spellings that DO collide with
/// standard organic labels (CA = C-alpha, CD = C-delta, HG = H-gamma,
/// PB = P-beta, and HO = hydroxyl H in AMBER) are deliberately absent —
/// their ionic meaning is expressed through ION_RESNAMES instead, exactly
/// the (resname, atomname) disambiguation GROMACS' atomprop database uses.
pub(crate) const ELEMENTAL_ATOM_NAMES: &[&str] = &[
    "FE", "ZN", "MN", "CU", "NI", "MG", "SE", "BR", "CL", "NA", "LI", "RB", "BA", "SR",
    "AL", "SI", "SN", "TI", "CR", "MO", "W", "AU", "AG", "PT", "PD", "GA", "GE", "AS",
    "SM", "GD", "YB", "EU", "TB", "DY", "ER", "TM", "LU", "NB", "ZR", "RU", "RH",
    "OS", "IR", "BE", "CO",
];

fn norm_element(s: &str) -> String {
    let u = s.trim().to_ascii_uppercase();
    if u.is_empty() {
        return String::from("X");
    }
    if TWO_LETTER_ELEMENTS.contains(&u.as_str()) {
        u
    } else {
        u.chars().next().map(|c| c.to_string()).unwrap_or_else(|| "X".into())
    }
}

/// Canonical capitalisation for output ("NA" -> "Na", "CL" -> "Cl").
fn canon_element(s: &str) -> String {
    let u = norm_element(s);
    let mut chars = u.chars();
    match (chars.next(), chars.next()) {
        (Some(a), Some(b)) => format!("{}{}", a, b.to_ascii_lowercase()),
        (Some(a), _) => a.to_string(),
        _ => String::from("X"),
    }
}

/// Shared element inference from an atom name (+ residue context).  Used by
/// BOTH the PDB template reader and the GRO topology reader so the two sides
/// can never disagree.  Priority follows the production tools: an explicit
/// element column wins (checked by the caller), then ion residues carry full
/// element symbols, then unambiguous elemental spellings (HEM/FE, MSE/SE,
/// ligand CL/BR), then organic atom-label conventions.  Returns the
/// canonical capitalisation ("Na", "Cl", "Fe"), matching [geom] conventions.
pub(crate) fn element_from_name(name_raw: &str, resname: &str) -> String {
    let up_res = resname.trim().to_ascii_uppercase();
    let up_name = name_raw.trim().to_ascii_uppercase();
    // letters-only prefix of the atom name (handles 1HB, CL1, CD1, O5', ...)
    let letters: Vec<char> = name_raw
        .trim()
        .trim_start_matches(|c: char| !c.is_ascii_alphabetic())
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    let letters_up: String = letters.iter().map(|c| c.to_ascii_uppercase()).collect();
    // 1) ion/metal residues: the atom name (or the residue name) carries the
    //    element; try full name, then its two-letter prefix, then the
    //    residue name (CHARMM CAL/CAL, GROMACS SOD/NA, ...)
    if ION_RESNAMES.contains(&up_res.as_str()) {
        let el = norm_element(&up_name);
        if TWO_LETTER_ELEMENTS.contains(&el.as_str()) {
            return canon_element(&el);
        }
        let two: String = letters_up.chars().take(2).collect();
        if TWO_LETTER_ELEMENTS.contains(&two.as_str()) {
            return canon_element(&two);
        }
        let elr = norm_element(&up_res);
        if TWO_LETTER_ELEMENTS.contains(&elr.as_str()) {
            return canon_element(&elr);
        }
        return canon_element(&el);
    }
    // 2) unambiguous elemental spellings (also with trailing digits/charges)
    if ELEMENTAL_ATOM_NAMES.contains(&letters_up.as_str()) {
        return canon_element(&letters_up);
    }
    // 3) organic labels: a lowercase second letter ("Fe", "Cl") still denotes
    //    a two-letter element; uppercase spellings are Greek-position labels
    if letters.len() >= 2 && letters[1].is_lowercase() {
        return canon_element(&letters_up[..2]);
    }
    // 4) fallback: first alphabetic character
    letters
        .first()
        .map(|c| c.to_ascii_uppercase().to_string())
        .unwrap_or_else(|| String::from("X"))
}

fn infer_element(name_raw: &str, resname: &str, elems_raw: &str) -> String {
    // trust an explicit element column verbatim (GROMACS/OpenMM behaviour)
    let from_col = elems_raw.trim();
    if !from_col.is_empty() {
        return norm_element(from_col);
    }
    element_from_name(name_raw, resname)
}

pub fn read_pdb_template(path: &str) -> std::io::Result<PdbTemplate> {
    let bytes = std::fs::read(path)?;
    let mut t = PdbTemplate {
        names_raw: Vec::new(),
        resnames: Vec::new(),
        resids: Vec::new(),
        chains: Vec::new(),
        elements: Vec::new(),
        elems_raw: Vec::new(),
        solvent: Vec::new(),
        positions: Vec::new(),
    };
    let mut seen_model = false;
    for line in bytes.split(|b| *b == b'\n') {
        let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
        let rec = field_str(line, 0, 6);
        let rec = rec.trim();
        if rec == "ENDMDL" {
            break; // first model only
        }
        if rec == "MODEL" {
            if seen_model {
                break;
            }
            seen_model = true;
            continue;
        }
        if rec != "ATOM" && rec != "HETATM" {
            continue;
        }
        let alt = field_str(line, 16, 17);
        let alt = alt.trim();
        if !alt.is_empty() && alt != "A" {
            continue; // skip alternate conformations other than 'A'
        }
        let name_raw = field_str(line, 12, 16);
        let resname = field_str(line, 17, 20).trim().to_string();
        let chain = format!("{:<1}", field_str(line, 21, 22));
        let elems_raw = field_str(line, 76, 78);
        let el = infer_element(&name_raw, &resname, &elems_raw);
        let x: f64 = field_str(line, 30, 38).trim().parse().unwrap_or(f64::NAN);
        let y: f64 = field_str(line, 38, 46).trim().parse().unwrap_or(f64::NAN);
        let z: f64 = field_str(line, 46, 54).trim().parse().unwrap_or(f64::NAN);
        t.names_raw.push(name_raw);
        t.resnames.push(resname.clone());
        t.resids.push(field_str(line, 22, 26).trim().to_string());
        t.chains.push(chain);
        t.elements.push(el);
        t.elems_raw.push(elems_raw);
        t.solvent
            .push(WATER_RESNAMES.contains(&resname.to_ascii_uppercase().as_str()));
        t.positions.push([x, y, z]);
    }
    Ok(t)
}

/// Map each system atom onto a template atom (nearest same-element within
/// 1.5 A of the initial coordinates).  Unmatched atoms (e.g. link hydrogens)
/// get `None` and are written with synthesized metadata.
pub fn map_pdb_template(
    tmpl: &PdbTemplate,
    symbols: &[String],
    pos_ang: &[f64],
) -> Vec<Option<usize>> {
    use std::collections::HashMap;
    let cell = 2.5f64;
    let key = |p: [f64; 3]| {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in tmpl.positions.iter().enumerate() {
        if p.iter().all(|v| v.is_finite()) {
            grid.entry(key(*p)).or_default().push(i);
        }
    }
    let mut out = vec![None; symbols.len()];
    for i in 0..symbols.len() {
        let el = norm_element(&symbols[i]);
        let p = [pos_ang[3 * i], pos_ang[3 * i + 1], pos_ang[3 * i + 2]];
        let (kx, ky, kz) = key(p);
        let mut best: Option<(f64, usize)> = None;
        for r in 0..3i64 {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if r > 0 && dx.abs() != r && dy.abs() != r && dz.abs() != r {
                            continue;
                        }
                        if let Some(cands) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &j in cands {
                                if tmpl.elements[j] != el {
                                    continue;
                                }
                                let q = tmpl.positions[j];
                                let d2 = (p[0] - q[0]).powi(2)
                                    + (p[1] - q[1]).powi(2)
                                    + (p[2] - q[2]).powi(2);
                                if best.map_or(true, |(bd, _)| d2 < bd) {
                                    best = Some((d2, j));
                                }
                            }
                        }
                    }
                }
            }
            if let Some((bd, _)) = best {
                if bd.sqrt() < cell * (r as f64) {
                    break;
                }
            }
        }
        if let Some((d2, j)) = best {
            if d2.sqrt() <= 1.5 {
                out[i] = Some(j);
            }
        }
    }
    out
}

fn pad_or_trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect::<String>()
}

/// Write one trajectory frame as a PDB MODEL, reproducing the reference
/// record layout (atom-name and element columns are copied verbatim from the
/// template; synthesized atoms follow the fixed-column conventions).
#[allow(clippy::too_many_arguments)]
pub fn write_pdb_frame<W: Write>(
    file: &mut W,
    tmpl: &PdbTemplate,
    mapping: &[Option<usize>],
    symbols: &[String],
    pos_bohr: &[f64],
    step: usize,
    box_ang: Option<[f64; 3]>,
    skip_solvent: bool,
) -> std::io::Result<()> {
    writeln!(file, "MODEL     {:>4}", step)?;
    if let Some(b) = box_ang {
        writeln!(
            file,
            "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} P 1           1",
            b[0], b[1], b[2], 90.0, 90.0, 90.0
        )?;
    }
    for i in 0..symbols.len() {
        let tm = mapping.get(i).copied().flatten();
        if skip_solvent && tm.map_or(false, |j| tmpl.solvent[j]) {
            continue;
        }
        let x = pos_bohr[3 * i] * crate::constants::BOHR;
        let y = pos_bohr[3 * i + 1] * crate::constants::BOHR;
        let z = pos_bohr[3 * i + 2] * crate::constants::BOHR;
        let (name_col, resn, resid, chain, elem_col) = match tm {
            Some(j) => (
                name_field(&tmpl.names_raw[j]),
                tmpl.resnames[j].clone(),
                tmpl.resids[j].clone(),
                tmpl.chains[j].clone(),
                elem_field(&tmpl.elements[j]),
            ),
            None => (
                name_field(&symbols[i]),
                String::from("UNK"),
                String::from("0"),
                String::from(" "),
                elem_field(&symbols[i]),
            ),
        };
        writeln!(
            file,
            "ATOM  {:5} {} {:>3} {}{:>4}    {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}          {}",
            (i + 1) % 100000,
            name_col,
            resn.chars().take(3).collect::<String>(),
            chain,
            resid.chars().take(4).collect::<String>(),
            x,
            y,
            z,
            1.0,
            0.0,
            elem_col
        )?;
    }
    writeln!(file, "ENDMDL")?;
    Ok(())
}

/// 4-character atom-name field: keep the template text when present,
/// otherwise right-align a short name (columns 14-16) as PDB does.
fn name_field(src: &str) -> String {
    if src.len() == 4 && !src.trim().is_empty() {
        return src.to_string();
    }
    let s = src.trim();
    match s.chars().count() {
        0 => String::from("    "),
        1 => format!(" {}  ", s),
        2 => format!(" {} ", s),
        3 => format!(" {} ", s),
        _ => s.chars().take(4).collect(),
    }
}

/// 2-character element field (columns 77-78), right-justified canonical
/// symbol — always normalised rather than copied verbatim, so files written
/// by tools with sloppy element columns still come out consistent.
fn elem_field(canonical: &str) -> String {
    format!("{:>2}", canon_element(canonical))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_protein_labels() {
        // protein atom labels: only the first letter is the element
        for (name, res, el) in [
            ("CA", "ALA", "C"),  // alpha carbon, NOT calcium
            ("CD", "ILE", "C"),  // C-delta
            ("CD1", "LEU", "C"),
            ("HG", "SER", "H"),  // H-gamma
            ("PB", "ATP", "P"),  // P-beta of a nucleotide
            ("NE2", "HIS", "N"),
            ("OD1", "ASP", "O"),
            ("SG", "CYS", "S"),
            ("1HB", "ALA", "H"), // CHARMM leading-digit hydrogens
            ("O5'", "ATP", "O"),
            ("C11", "HPAT", "C"),
            ("HO", "SER", "H"),  // AMBER hydroxyl hydrogen, not holmium
        ] {
            assert_eq!(element_from_name(name, res), el, "{}/{}", name, res);
        }
    }

    #[test]
    fn element_ion_residues() {
        for (name, res, el) in [
            ("NA", "NA", "Na"), ("NA", "SOD", "Na"), ("CL", "CLA", "Cl"),
            ("MG", "MG", "Mg"), ("CA", "CA", "Ca"), ("CAL", "CAL", "Ca"),
            ("K", "K", "K"), ("K", "POT", "K"), ("ZN", "ZN", "Zn"),
            ("CU1", "CU1", "Cu"), ("IOD", "IOD", "I"),
        ] {
            assert_eq!(element_from_name(name, res), el, "{}/{}", name, res);
        }
    }

    #[test]
    fn element_unambiguous_names() {
        // elemental spellings that are never organic labels
        for (name, res, el) in [
            ("FE", "HEM", "Fe"),  // heme iron, not fluorine
            ("SE", "MSE", "Se"),  // selenomethionine
            ("CL1", "BENZ", "Cl"), // numbered ligand halogen
            ("BR", "BNZ", "Br"),
            ("ZN", "ENZ", "Zn"),
        ] {
            assert_eq!(element_from_name(name, res), el, "{}/{}", name, res);
        }
    }

    #[test]
    fn element_lowercase_two_letter() {
        assert_eq!(element_from_name("Fe", "METAL"), "Fe");
        assert_eq!(element_from_name("Cl", "ION"), "Cl");
    }

    #[test]
    fn template_reader_uses_element_column() {
        let dir = std::env::temp_dir().join("rest_pdb_el_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.pdb");
        std::fs::write(
            &path,
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
             HETATM    2  FE  HEM A   2       1.000   0.000   0.000  1.00  0.00          Fe\n\
             HETATM    3  OW  SOL W   3       2.000   0.000   0.000  1.00  0.00           O\n\
             END\n",
        )
        .unwrap();
        let t = read_pdb_template(path.to_str().unwrap()).unwrap();
        assert_eq!(t.elements, vec!["C", "FE", "O"]);
        assert_eq!(t.solvent, vec![false, false, true]);
    }
}
