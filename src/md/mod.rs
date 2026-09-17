pub mod parameters;
pub mod integrator;
pub mod umbrella;
pub mod io;
pub mod mm;
pub mod mm_build;
pub mod ase;

use std::io::Write;

use tensors::MatrixFull;

use crate::mpi_io::MPIOperator;
use crate::scf_io::SCF;
use crate::utilities;
use integrator::{HARTREE2EV, AU_FORCE2_EV_PER_ANG, KB_HARTREE_PER_K};
use umbrella::UmbrellaBias;

struct ForceEval {
    epot_qm: f64,
    epot_mm: f64,
    epot_bias: f64,

    epot: f64,

    force: Vec<f64>,

    cv_values: Option<Vec<f64>>,

    dipole: Option<[f64; 3]>,
    max_f_qm_ev: f64,
    max_f_mm_ev: f64,
}

struct QmmmRuntime {
    n_qm_geom: usize,
    n_link: usize,
    gro_skip: usize,
    n_gro: usize,
    mm_start: usize,
    charges: Vec<f64>,
    box_nm: [f64; 3],
    mm_cutoff: f64,
    pbc: bool,
    xml_mode: bool,
}

impl QmmmRuntime {
    fn gro_row(&self, j: usize) -> usize {
        if j < self.gro_skip {
            j
        } else {
            self.mm_start + (j - self.gro_skip)
        }
    }
    fn gro_order_positions(&self, pos: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(3 * self.n_gro);
        for j in 0..self.n_gro {
            let r = 3 * self.gro_row(j);
            out.extend_from_slice(&pos[r..r + 3]);
        }
        out
    }
}

fn evaluate_forces(
    scf_data: &mut SCF,
    time_mark: &mut utilities::TimeRecords,
    mpi_operator: &Option<MPIOperator>,
    pos: &[f64],
    is_root: bool,
    n_qm_geom: usize,
    rt: Option<&QmmmRuntime>,
    engine: &mut Option<mm::OpenMmEngine>,
    qmmm_params: Option<&parameters::QmmmParams>,
    bias: Option<&UmbrellaBias>,
    want_dipole: bool,
) -> ForceEval {
    let n_total = pos.len() / 3;
    let mut epot_mm = 0.0f64;
    let mut f_mm: Vec<f64> = Vec::new();
    let mut sites: Vec<mm::EmbeddingSite> = Vec::new();

    if let (Some(rt), Some(qp)) = (rt, qmmm_params) {
        let gro_pos = if rt.xml_mode {
            rt.gro_order_positions(pos)
        } else {

            let mut gp = pos[..3 * n_qm_geom].to_vec();
            gp.extend_from_slice(&rt.gro_order_positions(pos));
            gp
        };
        let mut buf: Vec<f64> = Vec::new();
        if is_root {
            let (e_mm, force_mm) = engine
                .as_ref()
                .expect("OpenMM engine missing on the master rank")
                .evaluate(&gro_pos)
                .expect("OpenMM MM evaluation failed");
            buf.push(e_mm);
            buf.extend_from_slice(&force_mm);
        }
        bcast_f64(&mut buf, 0, mpi_operator);
        epot_mm = buf[0];
        f_mm = buf[1..].to_vec();

        sites = mm::build_embedding(
            &pos[..3 * n_qm_geom],
            &pos[3 * rt.mm_start..],
            &rt.charges,
            rt.box_nm,
            qp.pbc,
            rt.mm_cutoff,
        );
        let mut ghost_pos = Vec::with_capacity(3 * sites.len());
        let mut ghost_chg = Vec::with_capacity(sites.len());
        for s in sites.iter() {
            ghost_chg.push(s.charge);
            ghost_pos.extend_from_slice(&s.pos_bohr);
        }
        scf_data.mol.geom.ghost_pc_chrg = ghost_chg;
        scf_data.mol.geom.ghost_pc_pos = if sites.is_empty() {
            MatrixFull::empty()
        } else {
            MatrixFull::from_vec([3, sites.len()], ghost_pos).unwrap()
        };
    }

    let (epot_qm, grad_opt) = if n_qm_geom == 0 {
        (0.0, None)
    } else {
        let qm_matrix = MatrixFull::from_vec([3, n_qm_geom], pos[..3 * n_qm_geom].to_vec()).unwrap();
        let (e, g) =
            crate::main_driver::eval_force_with_position(scf_data, time_mark, mpi_operator, &qm_matrix);
        (e, Some(g))
    };

    let mut force = vec![0.0f64; 3 * n_total];
    if !f_mm.is_empty() {
        if let Some(rt) = rt {
            if rt.xml_mode {
                for j in 0..rt.n_gro {
                    let row = 3 * rt.gro_row(j);
                    for c in 0..3 {
                        force[row + c] += f_mm[3 * j + c];
                    }
                }
            } else {

                for i in 0..n_qm_geom {
                    for c in 0..3 {
                        force[3 * i + c] += f_mm[3 * i + c];
                    }
                }
                for j in rt.gro_skip..rt.n_gro {
                    let row = 3 * rt.gro_row(j);
                    for c in 0..3 {
                        force[row + c] += f_mm[3 * (n_qm_geom + j) + c];
                    }
                }
            }
        }
    }
    if let Some(gradient) = &grad_opt {
        let grad = gradient.data();
        for i in 0..3 * n_qm_geom {
            force[i] -= grad[i];
        }
    }
    if !sites.is_empty() {
        if let Some(ghost_forces) = scf_data.compute_ghost_charge_forces() {
            let gf = ghost_forces.data();
            let mm_start = rt.map(|r| r.mm_start).unwrap_or(n_qm_geom);
            let verbose = scf_data.mol.ctrl.print_level >= 2;
            if verbose {
                println!("------ Forces on embedding point charges [a.u.] ------");
                println!("      (site k -> MM atom #global_index)");
            }
            for (k, s) in sites.iter().enumerate() {
                let row = 3 * (mm_start + s.mm_index);
                if verbose {
                    println!(
                        "    Q{:04} (MM atom {:5})   {:15.8}{:15.8}{:15.8}",
                        k + 1,
                        s.mm_index + 1,
                        gf[3 * k],
                        gf[3 * k + 1],
                        gf[3 * k + 2]
                    );
                }
                for c in 0..3 {
                    force[row + c] += gf[c + 3 * k];
                }
            }
            if verbose {
                println!("------------------------------------------------------");
            }
        }
    }

    let mut epot_bias = 0.0f64;
    let mut cv_values = None;
    if let Some(b) = bias {
        let (e_b, f4) = b.energy_and_forces(pos);
        epot_bias = e_b;
        cv_values = Some(b.display_values(pos));
        for (k, f_atom) in f4.iter().enumerate() {
            for c in 0..3 {
                force[3 * b.atoms[k] + c] += f_atom[c];
            }
        }
    }

    if let Some(rt) = rt {
        for k in 0..rt.n_link {
            let row = rt.n_qm_geom - rt.n_link + k;
            for c in 0..3 {
                force[3 * row + c] = 0.0;
            }
        }
    }

    if let Some(bad) = force.iter().position(|x| !x.is_finite()) {
        panic!(
            "MD run: non-finite force component at combined atom {} — aborting before the \
             garbage force can corrupt the trajectory (check for overlapping particles or \
             an SCF divergence)",
            bad / 3
        );
    }
    let fmax = |rows: std::ops::Range<usize>| -> f64 {
        rows.map(|i| {
            let mut m = 0.0f64;
            for c in 0..3 {
                m = m.max(force[3 * i + c].abs());
            }
            m * AU_FORCE2_EV_PER_ANG
        })
        .fold(0.0f64, f64::max)
    };
    let max_f_qm_ev = fmax(0..n_qm_geom);
    let max_f_mm_ev = fmax(n_qm_geom..n_total);

    let dipole = if want_dipole {
        Some(crate::post_scf_analysis::evaluate_dipole_moment(&*scf_data, None))
    } else {
        None
    };

    let epot = epot_qm + epot_mm + epot_bias;
    ForceEval {
        epot_qm,
        epot_mm,
        epot_bias,
        epot,
        force,
        cv_values,
        dipole,
        max_f_qm_ev,
        max_f_mm_ev,
    }
}

#[cfg(feature = "mpi")]
fn bcast_f64(vec: &mut Vec<f64>, root: usize, mpi_operator: &Option<MPIOperator>) {
    if let Some(op) = mpi_operator {
        crate::mpi_io::mpi_broadcast_vector(&op.world, vec, root);
    }
}
#[cfg(not(feature = "mpi"))]
fn bcast_f64(_vec: &mut Vec<f64>, _root: usize, _mpi_operator: &Option<MPIOperator>) {}

pub fn is_pure_mm_run(ctrl_file: &str) -> bool {
    let raw = std::fs::read_to_string(ctrl_file).unwrap_or_default();
    let keys: serde_json::Value = if let Ok(v) = serde_json::from_str(&raw) {
        v
    } else {
        match toml::from_str::<serde_json::Value>(&raw) {
            Ok(v) => v,
            Err(_) => return false,
        }
    };
    keys.get("ctrl")
        .and_then(|c| c.get("pure_mm"))
        .map(|v| match v {
            serde_json::Value::Bool(b) => *b,
            serde_json::Value::String(s) => {
                s.eq_ignore_ascii_case("true") || s == "1" || s.eq_ignore_ascii_case("yes")
            }
            serde_json::Value::Number(n) => n.as_i64().map(|i| i != 0).unwrap_or(false),
            _ => false,
        })
        .unwrap_or(false)
}

pub fn run_pure_mm(ctrl_file: &str) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let raw = std::fs::read_to_string(ctrl_file)?;
    let keys: serde_json::Value = if let Ok(v) = serde_json::from_str(&raw) {
        v
    } else {
        toml::from_str::<serde_json::Value>(&raw)?
    };
    let params = parameters::parse_md_keywords(&keys)
        .unwrap_or_else(|e| panic!("MD run: invalid [md] section: {}", e))
        .unwrap_or_default();
    let job_type = keys
        .get("ctrl")
        .and_then(|c| c.get("job_type"))
        .and_then(|x| x.as_str())
        .unwrap_or("md")
        .to_lowercase();
    let is_single_point = job_type.eq("sp")
        || job_type.eq("force")
        || job_type.eq("singlepoint")
        || (job_type.eq("md") && params.ensemble.eq("sp"));
    if !job_type.eq("md") && !is_single_point {
        panic!(
            "pure_mm run: unsupported job_type '{}' (expected md|sp|force|singlepoint)",
            job_type
        );
    }

    let mm_file = params
        .qmmm
        .as_ref()
        .map(|q| q.mm_file.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| panic!("pure_mm run requires qmmm_mm_file"));
    let wall = std::time::Instant::now();

    let base = std::path::Path::new(ctrl_file).parent().unwrap_or(std::path::Path::new("."));
    let top = mm::parse_gro(base.join(&mm_file).to_str().unwrap())?;

    let lj_table = params.qmmm.as_ref().map(|q| q.lj.clone()).unwrap_or_default();

    let engine = if !params
        .qmmm
        .as_ref()
        .map(|q| q.system_xml.clone())
        .unwrap_or_default()
        .is_empty()
    {
        let xml_path = base.join(
            params.qmmm.as_ref().map(|q| q.system_xml.clone()).unwrap_or_default(),
        );
        let sys = mm::parse_openmm_system_xml(xml_path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("MD run: {}", e));
        if sys.n_particles != top.n_mm {
            panic!(
                "Pure-MM run: system XML has {} particles but the GRO file has {} atoms",
                sys.n_particles, top.n_mm
            );
        }
        mm::OpenMmEngine::new_from_system_xml(xml_path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("MD run: {}", e))
    } else {
        mm::OpenMmEngine::new(
            &[],
            &[],
            &lj_table,
            &top,
            params.qmmm.as_ref().map(|q| q.pbc).unwrap_or(true),
            0,
        )
        .unwrap_or_else(|e| panic!("MD run: {}", e))
    };
    let engine_ref = &engine;

    let n = top.n_mm;
    let mut pos: Vec<f64> = top.pos_bohr.clone();
    let symbols: Vec<String> = top.symbols.clone();
    let mass_amu: Vec<f64> = top.masses.clone();
    let mass_au = integrator::masses_amu_to_au(&mass_amu);

    println!(
        "Pure-MM run: {} particles from {}, engine = OpenMM (SOL residues typed as TIP3P)",
        n, mm_file
    );

    let evaluate = |pos_bohr: &[f64]| -> anyhow::Result<(f64, Vec<f64>)> {
        let (e_ha, f_au) = engine_ref.evaluate(pos_bohr)?;
        if f_au.iter().any(|x| !x.is_finite()) {
            panic!("Pure-MM run: non-finite forces — check for overlapping particles");
        }
        Ok((e_ha, f_au))
    };
    let last_e = std::cell::Cell::new(0.0f64);
    let last_f: std::rc::Rc<std::cell::RefCell<Vec<f64>>> =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let f_slot = last_f.clone();
    let mut hook = |positions_ang: &[f64]| -> anyhow::Result<(f64, Vec<f64>)> {
        let pos_bohr: Vec<f64> =
            positions_ang.iter().map(|x| x / crate::constants::BOHR).collect();
        let (e_ha, f_au) = evaluate(&pos_bohr)?;
        last_e.set(e_ha);
        *f_slot.borrow_mut() = f_au.clone();
        Ok((e_ha * HARTREE2EV, f_au.iter().map(|x| x * AU_FORCE2_EV_PER_ANG).collect()))
    };

    if is_single_point {
        let (e_ha, f_au) = evaluate(&pos)?;
        let fmax_au = f_au.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        println!("Potential Energy = {:.9} a.u.", e_ha);
        println!("Max force = {:.9} a.u.", fmax_au);
        println!("------ Output forces [a.u.] ------");
        for i in 0..n {
            println!(
                "{:<2}{:16.9}{:16.9}{:16.9}",
                symbols[i],
                f_au[3 * i],
                f_au[3 * i + 1],
                f_au[3 * i + 2]
            );
        }
        println!("--------------------------------------");
        return Ok(());
    }

    if params.ensemble.eq("opt") {
        let opt = ase::AseOpt::new(
            &symbols,
            &mass_amu,
            &pos.iter().map(|x| x * crate::constants::BOHR).collect::<Vec<f64>>(),
            &params.opt_algorithm,
            &[],
        )
        .unwrap_or_else(|e| panic!("MD run: {}", e));
        println!(
            "Pure-MM geometry optimization: {} (fmax = {} eV/A, max {} steps)",
            params.opt_algorithm, params.opt_fmax, params.opt_steps
        );
        let (p_ang, _e_ev, nsteps, converged) = opt
            .run(&mut hook, params.opt_fmax, params.opt_steps)
            .unwrap_or_else(|e| panic!("MD run: {}", e));
        for i in 0..3 * n {
            pos[i] = p_ang[i] / crate::constants::BOHR;
        }
        let (_, f_au) = evaluate(&pos)?;
        let fmax = f_au.iter().fold(0.0f64, |a, x| a.max(x.abs())) * AU_FORCE2_EV_PER_ANG;
        let mut f = std::fs::File::create("opt_mm.log").unwrap();
        writeln!(f, "# steps converged fmax_eV/A E_MM_eV").unwrap();
        writeln!(
            f,
            "{} {} {:.6} {:.9}",
            nsteps,
            converged,
            fmax,
            last_e.get() * HARTREE2EV
        )
        .unwrap();
        let mut s = format!("{}\n", n);
        for i in 0..n {
            s.push_str(&format!(
                "{:<2}{:15.8}{:15.8}{:15.8}\n",
                symbols[i],
                pos[3 * i] * crate::constants::BOHR,
                pos[3 * i + 1] * crate::constants::BOHR,
                pos[3 * i + 2] * crate::constants::BOHR
            ));
        }
        let _ = std::fs::write("opt_final.xyz", s);
        println!(
            "Pure-MM optimization done after {} force steps (converged = {}): opt_final.xyz / opt_mm.log, {:.1} s",
            nsteps,
            converged,
            wall.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    let mut log = std::fs::File::create("md.log").unwrap();
    writeln!(
        log,
        "#\"Step\",\"Time (ps)\",\"Potential Energy (kJ/mole)\",\"Kinetic Energy (kJ/mole)\",\"Total Energy (kJ/mole)\",\"Temperature (K)\""
    )
    .unwrap();
    let mut traj = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("dump_mm.xyz")
        .unwrap();
    let md = ase::AseMd::new(
        &symbols,
        &mass_amu,
        &pos.iter().map(|x| x * crate::constants::BOHR).collect::<Vec<f64>>(),
        None,
        params.temperature,
        params.friction,
        &params.friction_units,
        params.dt,
        &params.ensemble,
        42,
        &[],
    )
    .unwrap_or_else(|e| panic!("MD run: {}", e));
    let mut vel: Vec<f64> = md
        .velocities_ang_fs()
        .iter()
        .map(|v| v * integrator::AU_TIME_TO_FS / crate::constants::BOHR)
        .collect();
    let f_for_traj = last_f.clone();
    for step in 1..=params.steps {
        let (p_ang, v_ang) = md.step(&mut hook).unwrap_or_else(|e| panic!("MD run: {}", e));
        for i in 0..3 * n {
            pos[i] = p_ang[i] / crate::constants::BOHR;
            vel[i] = v_ang[i] * integrator::AU_TIME_TO_FS / crate::constants::BOHR;
        }
        let ekin = integrator::kinetic_energy_hartree(&vel, &mass_au);
        let epot = last_e.get();
        let temp = 2.0 * ekin / (3.0 * n as f64 * KB_HARTREE_PER_K);
        if step > params.equil_steps {
            let epot_kj = epot * crate::constants::HARTREE2KJMOL;
            let ekin_kj = ekin * crate::constants::HARTREE2KJMOL;
            let time_ps = step as f64 * params.dt / 1000.0;
            writeln!(
                log,
                "{:10.0},{:10.4},{:16.4},{:14.4},{:16.4},{:12.4}",
                step,
                time_ps,
                epot_kj,
                ekin_kj,
                epot_kj + ekin_kj,
                temp
            )
            .unwrap();
            if step % params.traj_interval.max(1) == 0 || step == params.steps {
                io::write_traj_frame(
                    &mut traj,
                    &symbols,
                    &pos,
                    &vel,
                    &f_for_traj.borrow(),
                    &mass_amu,
                    step - params.equil_steps,
                    epot,
                    None,
                    true,
                )
                .unwrap();
            }
        }
    }
    io::write_restart(
        std::path::Path::new("md_restart"),
        params.steps,
        params.dt,
        &symbols,
        &pos,
        &vel,
    )
    .unwrap();
    println!(
        "Pure-MM MD finished: {} steps in {:.1} s (md.log / dump_mm.xyz / md_restart)",
        params.steps,
        wall.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn run_md(
    scf_data: &mut SCF,
    time_mark: &mut utilities::TimeRecords,
    mpi_operator: &Option<MPIOperator>,
    ctrl_file: &str,
) {
    let is_root = mpi_operator.as_ref().map_or(true, |op| op.rank == 0);

    let raw = std::fs::read_to_string(ctrl_file).unwrap_or_else(|e| {
        panic!("MD run: cannot re-read the input file '{}': {}", ctrl_file, e)
    });
    let keys = if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
        v
    } else {
        toml::from_str::<serde_json::Value>(&raw)
            .unwrap_or_else(|e| panic!("MD run: cannot re-parse '{}': {}", ctrl_file, e))
    };
    let mut params = parameters::parse_md_keywords(&keys)
        .unwrap_or_else(|e| panic!("MD run: invalid [md] section: {}", e))
        .unwrap_or_default();
    if let Some(qp) = params.qmmm.as_mut() {
        if !qp.top.is_empty() {
            let base =
                std::path::Path::new(ctrl_file).parent().unwrap_or(std::path::Path::new("."));
            let qm_atoms: Vec<usize> = if qp.qm_atoms.is_empty() {
                (0..qp.gro_skip).collect()
            } else {
                qp.qm_atoms.clone()
            };
            let use_gro = qp.mm_pdb.is_empty() && !qp.mm_file.is_empty();
            let gro_path = if use_gro { qp.mm_file.clone() } else { String::new() };
            let n_atoms = mm_build::build_system_from_top(
                base.join(&qp.mm_pdb).to_str().unwrap(),
                base.join(&qp.top).to_str().unwrap(),
                base.join(&qp.ff_dir).to_str().unwrap(),
                &qm_atoms,
                &qp.links,
                &qp.frontier,
                &qp.frontier_rescale,
                qp.nb_cutoff,
                base.join("auto_system.xml").to_str().unwrap(),
                base.join("auto_mm.gro").to_str().unwrap(),
                &gro_path,
            )
            .unwrap_or_else(|e| panic!("MD run: {}", e));
            qp.gro_skip = qm_atoms.len();
            qp.system_xml = String::from("auto_system.xml");
            if !use_gro {
                qp.mm_file = String::from("auto_mm.gro");
            }
            if is_root {
                println!(
                    "QM/MM: built auto_system.xml from {} + {} ({} atoms, {} QM, rescale={}); gro = {}",
                    if qp.mm_pdb.is_empty() { "(external gro)" } else { &qp.mm_pdb },
                    qp.top,
                    n_atoms,
                    qm_atoms.len(),
                    qp.frontier_rescale,
                    qp.mm_file
                );
            }
        } else if qp.build_system {
            let base =
                std::path::Path::new(ctrl_file).parent().unwrap_or(std::path::Path::new("."));
            let pdb_path = base.join(&qp.mm_pdb);
            let n_atoms = mm::build_system_from_pdb(
                pdb_path.to_str().unwrap(),
                &qp.ff,
                &qp.qm_atoms,
                qp.gro_skip,
                &qp.links,
                &qp.frontier,
                &qp.frontier_rescale,
                qp.pbc,
                base.join("auto_system.xml").to_str().unwrap(),
                base.join("auto_mm.gro").to_str().unwrap(),
            )
            .unwrap_or_else(|e| panic!("MD run: {}", e));
            let qm_list: Vec<usize> = if qp.qm_atoms.is_empty() {
                (0..qp.gro_skip).collect()
            } else {
                qp.qm_atoms.clone()
            };
            let mut sorted_qm = qm_list.clone();
            sorted_qm.sort_unstable();
            sorted_qm.dedup();
            let mm_list: Vec<usize> =
                (0..n_atoms).filter(|i| !qm_list.contains(i)).collect();
            qp.links = qp
                .links
                .iter()
                .map(|(qb, mh)| {
                    let nq = sorted_qm
                        .iter()
                        .position(|x| x == qb)
                        .unwrap_or_else(|| panic!("MD run: link qm_bdry {} not in qmmm_qm_atoms", qb));
                    let nm = mm_list
                        .iter()
                        .position(|x| x == mh)
                        .unwrap_or_else(|| panic!("MD run: link mm_host {} is a QM atom", mh));
                    (nq, sorted_qm.len() + nm)
                })
                .collect();
            qp.gro_skip = sorted_qm.len();
            qp.system_xml = String::from("auto_system.xml");
            qp.mm_file = String::from("auto_mm.gro");
            if is_root {
                println!(
                    "QM/MM: built auto_system.xml + auto_mm.gro + auto_qm.xyz from {} with {:?} ({} atoms, {} QM, rescale={})",
                    qp.mm_pdb, qp.ff, n_atoms, qp.gro_skip, qp.frontier_rescale
                );
            }
        }
    }
    if is_root {
        println!("=========================================================");
        println!("{}", params.formated_report());
        println!("=========================================================");
    }

    let mut seed = params.seed;
    if seed == 0 {
        if is_root {
            seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345);
        }
        let mut buf = vec![f64::from_bits(seed)];
        bcast_f64(&mut buf, 0, mpi_operator);
        seed = buf[0].to_bits();
    }

    let mut symbols: Vec<String> = scf_data.mol.geom.elem.clone();
    let mut mass_amu: Vec<f64> =
        crate::geom_io::get_mass_charge(&scf_data.mol.geom.elem).iter().map(|(m, _)| *m).collect();
    let mut pos: Vec<f64> = scf_data.mol.geom.position.data();
    let n_qm = scf_data.mol.geom.elem.len();

    let mut mm_sys_owned: Option<mm::MmSystem> = None;
    let mut engine: Option<mm::OpenMmEngine> = None;
    let mut mm_charges: Vec<f64> = Vec::new();
    let mut gro_skip = 0usize;
    let mut xml_mode = false;
    let mut n_gro = 0usize;
    let mut box_nm = [0.0f64; 3];
    if let Some(qp) = params.qmmm.as_ref() {
        if !qp.system_xml.is_empty() && qp.gro_skip == 0 {
            panic!(
                "MD run: qmmm_system_xml requires qmmm_gro_skip > 0 (number of leading GRO atoms that duplicate the [geom] QM block)"
            );
        }
        gro_skip = qp.gro_skip;
        xml_mode = !qp.system_xml.is_empty();

        let base = std::path::Path::new(ctrl_file).parent().unwrap_or(std::path::Path::new("."));

        let top = if qp.build_box {
            if is_root {
                println!(
                    "QM/MM: building a TIP3P water box around the [geom] solute (margin = {:.1} A, spacing = {:.1} A)...",
                    qp.box_margin, qp.box_spacing
                );
            }
            let (gro_text, n_waters) = mm::build_solute_water_box(
                &scf_data.mol.geom.elem,
                &pos,
                qp.box_margin,
                qp.box_spacing,
                0,
            );
            let box_path = base.join("auto_box.gro");
            std::fs::write(&box_path, &gro_text)
                .unwrap_or_else(|e| panic!("MD run: cannot write auto_box.gro: {}", e));
            if is_root {
                println!("QM/MM: {} waters generated, box written to auto_box.gro", n_waters);
            }
            mm::parse_gro(box_path.to_str().unwrap()).unwrap_or_else(|e| panic!("MD run: {}", e))
        } else {
            let gro_path = base.join(&qp.mm_file);
            mm::parse_gro(gro_path.to_str().unwrap()).unwrap_or_else(|e| panic!("MD run: {}", e))
        };
        n_gro = top.n_mm + 0;
        box_nm = top.box_nm;
        if is_root {
            println!(
                "QM/MM: {} GRO atoms ({} skipped as QM duplicates), box {:.3} x {:.3} x {:.3} nm",
                top.n_mm, gro_skip, top.box_nm[0], top.box_nm[1], top.box_nm[2]
            );
        }
        if xml_mode {
            let xml_path = base.join(&qp.system_xml);
            let sys = mm::parse_openmm_system_xml(xml_path.to_str().unwrap())
                .unwrap_or_else(|e| panic!("MD run: {}", e));
            if sys.n_particles != n_gro {
                panic!(
                    "MD run: system xml has {} particles but the GRO file has {} atoms",
                    sys.n_particles, n_gro
                );
            }
            if gro_skip >= n_gro {
                panic!("MD run: qmmm_gro_skip = {} >= GRO atom count {}", gro_skip, n_gro);
            }
            if is_root {
                engine = Some(
                    mm::OpenMmEngine::new_from_system_xml(xml_path.to_str().unwrap())
                        .unwrap_or_else(|e| panic!("MD run: {}", e)),
                );
            }
            mm_sys_owned = Some(sys);
        } else {
            if is_root {
                engine = Some(
                    mm::OpenMmEngine::new(&symbols, &mass_amu, &qp.lj, &top, qp.pbc, gro_skip)
                        .unwrap_or_else(|e| panic!("MD run: {}", e)),
                );
            }
        }

        mm_charges = top.charges[gro_skip..].to_vec();
        symbols.extend_from_slice(&top.symbols[gro_skip..]);
        mass_amu.extend_from_slice(&top.masses[gro_skip..]);
        pos.extend_from_slice(&top.pos_bohr[3 * gro_skip..]);
    }
    let qmmm_params = params.qmmm.as_ref();
    let n_total = symbols.len();
    let mass_au = integrator::masses_amu_to_au(&mass_amu);

    let mut rt_owned: Option<QmmmRuntime> = params.qmmm.as_ref().map(|qp| {
        let n_link = qp.links.len();
        if n_link > n_qm {
            panic!(
                "MD run: {} link atoms requested but [geom] only has {} atoms (append the link H atoms to [geom])",
                n_link, n_qm
            );
        }
        for (qm_bdry, mm_host) in qp.links.iter() {
            if *qm_bdry >= n_qm {
                panic!(
                    "MD run: qmmm_links qm_bdry {} out of range ([geom] has {} atoms)",
                    qm_bdry, n_qm
                );
            }
            if *mm_host < gro_skip || *mm_host >= n_gro {
                panic!(
                    "MD run: qmmm_links mm_host {} out of range (GRO atoms {}..{})",
                    mm_host, gro_skip, n_gro
                );
            }
        }
        let n_link = qp.links.len();
        QmmmRuntime {
            n_qm_geom: n_qm,
            n_link,
            gro_skip,
            n_gro,
            mm_start: n_qm,
            charges: if xml_mode {
                mm_sys_owned.as_ref().unwrap().charges[gro_skip..].to_vec()
            } else {

                mm_charges.clone()
            },
            box_nm,
            mm_cutoff: qp.mm_cutoff,
            pbc: qp.pbc,
            xml_mode,
        }
    });
    let rt = rt_owned.as_ref();
    if let (Some(rt), Some(qp)) = (rt, params.qmmm.as_ref()) {
        if is_root && rt.n_link > 0 {
            for (k, (qm_bdry, mm_host)) in qp.links.iter().enumerate() {
                println!(
                    "Link H {:2}: geom atom {:3} (frozen), qm_bdry = {:3}, mm_host = GRO atom {:3}",
                    k + 1,
                    n_qm - rt.n_link + k,
                    qm_bdry,
                    mm_host
                );
            }
        }
    }

    let bias_owned: Option<UmbrellaBias> = params.umbrella.as_ref().map(|u| {
        if u.atoms.iter().any(|a| *a >= n_total) {
            panic!("MD run: umbrella_atoms {:?} out of range ({} atoms)", u.atoms, n_total);
        }
        UmbrellaBias::new(
            u.cv_type(),
            &u.atoms,
            u.center,
            u.kappa,
            u.umbrella_potential(),
            u.sum_center,
            u.sum_kappa,
        )
        .unwrap_or_else(|e| panic!("MD run: umbrella setup: {}", e))
    });
    let bias = bias_owned.as_ref();

    if let Some(op) = mpi_operator {
        if op.size > 1 {
            panic!(
                "MD run: multi-process MPI is not supported for MD; run single-process \
                 with OpenMP threads"
            );
        }
    }

    let restart_mode = !params.restart_input.is_empty();
    let mut vel: Vec<f64> = if restart_mode {
        let base = std::path::Path::new(ctrl_file).parent().unwrap_or(std::path::Path::new("."));
        let restart_path = base.join(&params.restart_input);
        let (rpos, rvel) = io::read_restart(restart_path.to_str().unwrap(), &symbols)
            .unwrap_or_else(|e| panic!("MD run: {}", e));
        if rpos.len() != 3 * n_total {
            panic!(
                "MD run: restart file '{}' has {} atoms but the system has {}",
                params.restart_input,
                rpos.len() / 3,
                n_total
            );
        }
        pos = rpos;
        if is_root {
            println!(
                "MD restart: positions and velocities read from {} ({} atoms)",
                params.restart_input, n_total
            );
        }
        rvel
    } else {
        match params.init_velocities.as_str() {
            "file" => io::read_velocities_file(&params.velocities_file, &symbols)
                .unwrap_or_else(|e| panic!("MD run: {}", e)),
            _ => vec![0.0; 3 * n_total],
        }
    };

    let ase_engine = {
        let fixed_rows: Vec<usize> = rt
            .map(|r| (0..r.n_link).map(|k| r.n_qm_geom - r.n_link + k).collect())
            .unwrap_or_default();
        let v_init: Option<Vec<f64>> = if restart_mode || params.init_velocities.eq("file") {
            let ang_fs = crate::constants::BOHR / integrator::AU_TIME_TO_FS;
            Some(vel.iter().map(|v| v * ang_fs).collect())
        } else if params.init_velocities.eq("zero") {
            Some(vec![0.0; 3 * n_total])
        } else {
            None
        };
        let pos_ang: Vec<f64> = pos.iter().map(|x| x * crate::constants::BOHR).collect();
        let engine = ase::AseMd::new(
            &symbols,
            &mass_amu,
            &pos_ang,
            v_init.as_deref(),
            params.temperature,
            params.friction,
            &params.friction_units,
            params.dt,
            &params.ensemble,
            seed,
            &fixed_rows,
        )
        .unwrap_or_else(|e| panic!("MD run: {}", e));

        vel = engine
            .velocities_ang_fs()
            .iter()
            .map(|v| v * integrator::AU_TIME_TO_FS / crate::constants::BOHR)
            .collect();
        let g_fs = if params.ensemble.eq("nvt") { params.friction_fs_inv() } else { 0.0 };
        println!(
            "MD engine: ASE {} (friction = {} {} -> {:.4} 1/fs, tau = {:.1} fs)",
            if params.ensemble.eq("nvt") { "Langevin" } else { "VelocityVerlet" },
            params.friction,
            params.friction_units,
            g_fs,
            if g_fs > 0.0 { 1.0 / g_fs } else { f64::INFINITY }
        );
        engine
    };

    let prefix = params.out_prefix.clone();
    let open = |name: String| -> Option<std::fs::File> {
        if is_root {
            Some(std::fs::File::create(name).unwrap_or_else(|e| panic!("MD run: {}", e)))
        } else {
            None
        }
    };
    let mut f_mdlog = if params.output_enabled("md_log") {
        open(format!("{}md.log", prefix))
    } else {
        None
    };
    let mut f_dip = if params.output_enabled("dipole") {
        open(format!("{}dipole.dat", prefix))
    } else {
        None
    };
    let mut f_traj = if params.output_enabled("traj") {
        open(format!("{}dump_{}.xyz", prefix, params.ensemble))
    } else {
        None
    };
    let mut f_efl = if params.qmmm.is_some() && params.output_enabled("energy_force") {
        let mut f = open(format!("{}energy_force.log", prefix));
        if let Some(f) = f.as_mut() {
            let _ = io::write_energy_force_header(f);
        }
        f
    } else {
        None
    };
    let mut f_umb = if params.umbrella.is_some() && params.output_enabled("umbrella_csv") {
        let mut f = open(format!("{}umbrella_timeseries.csv", prefix));
        if let Some(f) = f.as_mut() {
            let cols = bias_owned
                .as_ref()
                .map(|b| b.csv_columns())
                .unwrap_or_else(|| vec!["phi_deg"]);
            let _ = io::write_umbrella_header(f, &cols);
        }
        f
    } else {
        None
    };
    let write_restart_files = params.output_enabled("restart");

    let dt_fs = params.dt;
    let is_nvt = params.ensemble.eq("nvt");

    let wall_start = std::time::Instant::now();

    let update_links = |pos: &mut Vec<f64>, vel: &mut Vec<f64>, zero_v: bool| {
        if let (Some(rt), Some(qp)) = (rt, params.qmmm.as_ref()) {
            for (k, (qm_bdry, mm_host)) in qp.links.iter().enumerate() {
                let lrow = rt.n_qm_geom - rt.n_link + k;
                let qrow = 3 * qm_bdry;
                let mrow = 3 * rt.gro_row(*mm_host);
                let d = [
                    pos[mrow] - pos[qrow],
                    pos[mrow + 1] - pos[qrow + 1],
                    pos[mrow + 2] - pos[qrow + 2],
                ];
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let r_eq = qp.link_r_eq / crate::constants::BOHR;
                let unit = if n > 1.0e-12 {
                    [d[0] / n, d[1] / n, d[2] / n]
                } else {
                    [r_eq, 0.0, 0.0]
                };
                pos[3 * lrow] = pos[qrow] + r_eq * unit[0];
                pos[3 * lrow + 1] = pos[qrow] + r_eq * unit[1];
                pos[3 * lrow + 2] = pos[qrow] + r_eq * unit[2];
                if zero_v {
                    for c in 0..3 {
                        vel[3 * lrow + c] = 0.0;
                    }
                }
            }
        }
    };
    update_links(&mut pos, &mut vel, true);

    if params.ensemble.eq("sp") || params.ensemble.eq("singlepoint") {
        let fe = evaluate_forces(
            scf_data, time_mark, mpi_operator, &pos, is_root, n_qm, rt, &mut engine,
            qmmm_params, bias, f_dip.is_some(),
        );
        if is_root {
            let fmax =
                fe.force.iter().fold(0.0f64, |a, x| a.max(x.abs())) * AU_FORCE2_EV_PER_ANG;
            println!(
                "Single point: E_QM = {:.9} eV, E_MM = {:.9} eV, E_tot = {:.9} eV, max|F| = {:.6} eV/A",
                fe.epot_qm * HARTREE2EV,
                fe.epot_mm * HARTREE2EV,
                fe.epot * HARTREE2EV,
                fmax
            );
            if let Some(f) = f_efl.as_mut() {
                let _ = io::write_energy_force_line(
                    f,
                    1,
                    fe.epot_qm * HARTREE2EV,
                    fe.epot_mm * HARTREE2EV,
                    (fe.epot_qm + fe.epot_mm) * HARTREE2EV,
                    fe.max_f_qm_ev,
                    fe.max_f_mm_ev,
                );
            }
        }
        return;
    }

    if params.ensemble.eq("opt") {
        let last_feval = std::rc::Rc::new(std::cell::RefCell::new(None::<ForceEval>));
        let mut hook = |positions_ang: &[f64]| -> anyhow::Result<(f64, Vec<f64>)> {
            let mut pos_bohr: Vec<f64> =
                positions_ang.iter().map(|x| x / crate::constants::BOHR).collect();

            if let (Some(rt), Some(qp)) = (rt, params.qmmm.as_ref()) {
                for (k, (qm_bdry, mm_host)) in qp.links.iter().enumerate() {
                    let lrow = rt.n_qm_geom - rt.n_link + k;
                    let qrow = 3 * qm_bdry;
                    let mrow = 3 * rt.gro_row(*mm_host);
                    let d = [
                        pos_bohr[mrow] - pos_bohr[qrow],
                        pos_bohr[mrow + 1] - pos_bohr[qrow + 1],
                        pos_bohr[mrow + 2] - pos_bohr[qrow + 2],
                    ];
                    let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let r_eq = qp.link_r_eq / crate::constants::BOHR;
                    let unit = if dd > 1.0e-12 {
                        [d[0] / dd, d[1] / dd, d[2] / dd]
                    } else {
                        [1.0, 0.0, 0.0]
                    };
                    pos_bohr[3 * lrow] = pos_bohr[qrow] + r_eq * unit[0];
                    pos_bohr[3 * lrow + 1] = pos_bohr[qrow] + r_eq * unit[1];
                    pos_bohr[3 * lrow + 2] = pos_bohr[qrow] + r_eq * unit[2];
                }
            }
            let fe = evaluate_forces(
                scf_data, time_mark, mpi_operator, &pos_bohr, is_root, n_qm, rt,
                &mut engine, qmmm_params, bias, false,
            );
            if !fe.force.iter().all(|x| x.is_finite()) {
                panic!("MD run: non-finite forces during optimization — aborting");
            }
            let out = (
                fe.epot * HARTREE2EV,
                fe.force.iter().map(|x| x * AU_FORCE2_EV_PER_ANG).collect::<Vec<f64>>(),
            );
            *last_feval.borrow_mut() = Some(fe);
            Ok(out)
        };
        let n_link = rt.map(|r| r.n_link).unwrap_or(0);
        let frozen: Vec<usize> = (0..n_link)
            .map(|k| n_qm - n_link + k)
            .collect();
        let opt = ase::AseOpt::new(
            &symbols,
            &mass_amu,
            &pos.iter().map(|x| x * crate::constants::BOHR).collect::<Vec<f64>>(),
            &params.opt_algorithm,
            &frozen,
        )
        .unwrap_or_else(|e| panic!("MD run: {}", e));
        println!(
            "Geometry optimization: {} (fmax = {} eV/A, max {} steps), {} atoms",
            params.opt_algorithm, params.opt_fmax, params.opt_steps, n_total
        );
        let (p_ang, _e_ev, nsteps, converged) =
            opt.run(&mut hook, params.opt_fmax, params.opt_steps)
                .unwrap_or_else(|e| panic!("MD run: {}", e));
        for i in 0..3 * n_total {
            pos[i] = p_ang[i] / crate::constants::BOHR;
        }
        let vel_zero = vec![0.0f64; 3 * n_total];
        if write_restart_files {
            let _ = io::write_restart(
                std::path::Path::new(&format!("{}{}", params.out_prefix, "opt_restart")),
                params.steps, dt_fs, &symbols, &pos, &vel_zero,
            );
        }
        let n_ghost =
            if rt.is_some() { 0 } else { scf_data.mol.geom.ghost_pc_chrg.len() };
        let mut s = format!("{}\n", n_total + n_ghost);
        for i in 0..n_total {
            s.push_str(&format!(
                "{:<2}{:15.8}{:15.8}{:15.8}\n",
                symbols[i],
                pos[3 * i] * crate::constants::BOHR,
                pos[3 * i + 1] * crate::constants::BOHR,
                pos[3 * i + 2] * crate::constants::BOHR
            ));
        }

        let gpos = scf_data.mol.geom.ghost_pc_pos.data();
        let gelem: Vec<String> = scf_data.mol.geom.ghost_bs_elem.clone();
        for k in 0..n_ghost {
            let (sym, gx, gy, gz) = if k < gpos.len() / 3 {

                (
                    gelem.get(k).cloned().unwrap_or_else(|| "X".to_string()),
                    gpos[3 * k] * crate::constants::BOHR,
                    gpos[3 * k + 1] * crate::constants::BOHR,
                    gpos[3 * k + 2] * crate::constants::BOHR,
                )
            } else {
                ("X".to_string(), 0.0, 0.0, 0.0)
            };
            s.push_str(&format!("{:<2}{:15.8}{:15.8}{:15.8}\n", sym, gx, gy, gz));
        }
        let _ = std::fs::write(format!("{}opt_final.xyz", params.out_prefix), s);
        let fev = last_feval.borrow();
        let (epot_ev, fmax_ev) = match fev.as_ref() {
            Some(f) => (
                f.epot * HARTREE2EV,
                f.force.iter().fold(0.0f64, |a, x| a.max(x.abs())) * AU_FORCE2_EV_PER_ANG,
            ),
            None => (0.0, 0.0),
        };
        println!(
            "Geometry optimization done after {} force steps (converged = {}): \
             E = {:.6} eV, fmax = {:.6} eV/A; wrote {}opt_final.xyz / {}opt_restart",
            nsteps, converged, epot_ev, fmax_ev, params.out_prefix, params.out_prefix
        );
        return;
    }

    let last_feval = std::rc::Rc::new(std::cell::RefCell::new(None::<ForceEval>));
    let mut prev_pos = pos.clone();

    for step in 1..=params.steps {
        let mut hook = |positions_ang: &[f64]| -> anyhow::Result<(f64, Vec<f64>)> {
            let pos_bohr: Vec<f64> =
                positions_ang.iter().map(|x| x / crate::constants::BOHR).collect();
            let fe = evaluate_forces(
                scf_data, time_mark, mpi_operator, &pos_bohr, is_root, n_qm, rt,
                &mut engine, qmmm_params, bias, f_dip.is_some(),
            );
            if !scf_data.scf_converged {

                panic!(
                    "MD run: SCF did not converge at the current geometry — aborting to \
                     protect the trajectory (delete a stale 'restart' chkfile if present)"
                );
            }
            let out = (
                fe.epot * HARTREE2EV,
                fe.force.iter().map(|x| x * AU_FORCE2_EV_PER_ANG).collect::<Vec<f64>>(),
            );
            *last_feval.borrow_mut() = Some(fe);
            Ok(out)
        };
        let (p_ang, v_ang) =
            ase_engine.step(&mut hook).unwrap_or_else(|e| panic!("MD run: {}", e));
        for i in 0..3 * n_total {
            pos[i] = p_ang[i] / crate::constants::BOHR;
            vel[i] = v_ang[i] * integrator::AU_TIME_TO_FS / crate::constants::BOHR;
        }

        let link_lo = rt.map(|r| r.n_qm_geom - r.n_link).unwrap_or(0);
        let link_hi = rt.map(|r| r.n_qm_geom).unwrap_or(0);
        let mut worst = 0.0f64;
        let mut worst_atom = 0usize;
        for i in 0..n_total {
            if i >= link_lo && i < link_hi {
                continue;
            }
            let dx = pos[3 * i] - prev_pos[3 * i];
            let dy = pos[3 * i + 1] - prev_pos[3 * i + 1];
            let dz = pos[3 * i + 2] - prev_pos[3 * i + 2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > worst {
                worst = d;
                worst_atom = i;
            }
        }
        if worst * crate::constants::BOHR > 1.0 {
            panic!(
                "MD run: atom {} moved {:.3} A in a single {} step — corrupted dynamics, \
                 aborting (typical displacement at 300 K is < 0.05 A)",
                worst_atom,
                worst * crate::constants::BOHR,
                params.ensemble
            );
        }
        prev_pos.copy_from_slice(&pos);

        update_links(&mut pos, &mut vel, false);
        let feval = last_feval.borrow_mut().take().unwrap();

        let in_equil = step <= params.equil_steps;
        let pstep = step - params.equil_steps;
        if is_root && !in_equil {
            let ekin = integrator::kinetic_energy_hartree(&vel, &mass_au);
            let temp = 2.0 * ekin / (3.0 * n_total as f64 * KB_HARTREE_PER_K);
            let time_fs = pstep as f64 * dt_fs;
            let (epot_ev, ekin_ev, etot_ev) = (
                feval.epot * HARTREE2EV,
                ekin * HARTREE2EV,
                (feval.epot + ekin) * HARTREE2EV,
            );
            if let Some(f) = f_mdlog.as_mut() {
                let _ = io::write_md_log_line(
                    f, pstep, time_fs, epot_ev, ekin_ev, etot_ev, temp, feval.dipole,
                );
            }
            if let (Some(f), Some(d)) = (f_dip.as_mut(), feval.dipole) {
                let _ = io::write_dipole_line(f, d);
            }
            if let Some(f) = f_efl.as_mut() {
                let _ = io::write_energy_force_line(
                    f, step,
                    feval.epot_qm * HARTREE2EV,
                    feval.epot_mm * HARTREE2EV,
                    (feval.epot_qm + feval.epot_mm) * HARTREE2EV,
                    feval.max_f_qm_ev,
                    feval.max_f_mm_ev,
                );
            }
            if pstep % params.traj_interval == 0 || pstep == params.steps {
                if let Some(f) = f_traj.as_mut() {
                    let _ = io::write_traj_frame(
                        f, &symbols, &pos, &vel, &feval.force, &mass_amu,
                        pstep, feval.epot * HARTREE2EV, feval.dipole,
                        params.qmmm.is_some(),
                    );
                }
                if let (Some(f), Some(cv_vals)) = (f_umb.as_mut(), feval.cv_values.as_deref()) {
                    let _ = io::write_umbrella_line(
                        f, pstep, time_fs, cv_vals,
                        feval.epot_bias * HARTREE2EV, etot_ev, temp,
                    );
                }
                if write_restart_files {
                    let _ = io::write_restart(
                        std::path::Path::new(&format!("{}{}", prefix, params.restart_output)),
                        step, dt_fs, &symbols, &pos, &vel,
                    );
                }
            }
            if params.print_interval(step) {
                println!(
                    "MD step {:6} / {} : Epot = {:.6} eV, T = {:.2} K, elapsed {:.1} s",
                    pstep, params.steps, epot_ev, temp, wall_start.elapsed().as_secs_f64()
                );
            }
        }
    }

    if is_root {
        if write_restart_files {
            let _ = io::write_restart(
                std::path::Path::new(&format!("{}{}", prefix, params.restart_output)),
                params.steps, dt_fs, &symbols, &pos, &vel,
            );
        }
        println!(
            "MD finished: {} steps in {:.1} s; outputs: {}md.log, {}dump_{}.xyz, {}{}",
            params.steps,
            wall_start.elapsed().as_secs_f64(),
            prefix, prefix, params.ensemble, prefix, params.restart_output
        );
        let _ = std::io::stdout().flush();
    }
}
