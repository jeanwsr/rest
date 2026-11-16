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
