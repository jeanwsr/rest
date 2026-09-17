use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::integrator::{AU_FORCE2_EV_PER_ANG, EV2KJMOL};
use crate::constants::{BOHR, HARTREE2KJMOL};

#[derive(Debug, Clone, Default)]
pub struct MmSystem {
    pub n_particles: usize,
    pub charges: Vec<f64>,
    pub sigma_nm: Vec<f64>,
    pub eps_kjmol: Vec<f64>,
    pub masses: Vec<f64>,

    pub bonds: Vec<(usize, usize, f64, f64)>,

    pub angles: Vec<(usize, usize, usize, f64, f64)>,

    pub torsions: Vec<(usize, usize, usize, usize, f64, f64, u32)>,

    pub exceptions: Vec<(usize, usize)>,
    pub box_nm: [f64; 3],
}

fn xml_attr_f(node: roxmltree::Node, names: &[&str]) -> Option<f64> {
    for n in names {
        if let Some(v) = node.attribute(*n) {
            return v.parse().ok();
        }
    }
    None
}

pub fn parse_openmm_system_xml(path: &str) -> anyhow::Result<MmSystem> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read system xml '{}': {}", path, e))?;
    let doc = roxmltree::Document::parse(&content)
        .map_err(|e| anyhow::anyhow!("cannot parse system xml '{}': {}", path, e))?;
    let root = doc.root_element();
    if !root.has_tag_name("System") {
        return Err(anyhow::anyhow!("system xml '{}': root element is not <System>", path));
    }

    let mut sys = MmSystem { n_particles: 0, ..Default::default() };
    for node in root.children().filter(|n| n.is_element()) {
        match node.tag_name().name() {
            "Particles" | "Particle" => {
                let nodes: Vec<roxmltree::Node> = if node.has_tag_name("Particles") {
                    node.children()
                        .filter(|n| n.is_element() && n.has_tag_name("Particle"))
                        .collect()
                } else {
                    vec![node]
                };
                for p in nodes {
                    sys.masses.push(xml_attr_f(p, &["mass"]).unwrap_or(0.0));
                    sys.charges.push(0.0);
                    sys.sigma_nm.push(0.0);
                    sys.eps_kjmol.push(0.0);
                    sys.n_particles += 1;
                }
            }
            "Forces" | "Force" => {
                let nodes: Vec<roxmltree::Node> = if node.has_tag_name("Force") {
                    vec![node]
                } else {
                    node.children().filter(|n| n.is_element()).collect()
                };
                for force in nodes {
                    let ftype = force
                        .attribute("type")
                        .or_else(|| force.attribute("name"))
                        .unwrap_or("");
                    if ftype.contains("Nonbonded") {
                        for exc in force
                            .descendants()
                            .filter(|n| n.is_element() && n.has_tag_name("Exception"))
                        {
                            if let (Some(i), Some(j)) = (
                                xml_attr_f(exc, &["particle1", "p1"]).map(|v| v as usize),
                                xml_attr_f(exc, &["particle2", "p2"]).map(|v| v as usize),
                            ) {
                                if i < sys.n_particles && j < sys.n_particles {
                                    sys.exceptions.push((i.min(j), i.max(j)));
                                }
                            }
                        }
                        let mut nb_idx = 0usize;
                        for p in force.descendants().filter(|n| n.is_element() && n.has_tag_name("Particle")) {
                            match p.tag_name().name() {
                                "Particle" => {

                                    let i = nb_idx;
                                    nb_idx += 1;
                                    if let (Some(q), Some(s), Some(e)) = (
                                        xml_attr_f(p, &["q", "charge"]),
                                        xml_attr_f(p, &["sigma", "sig"]),
                                        xml_attr_f(p, &["epsilon", "eps"]),
                                    ) {
                                        if i < sys.n_particles {
                                            sys.charges[i] = q;
                                            sys.sigma_nm[i] = s;
                                            sys.eps_kjmol[i] = e;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else if ftype.contains("HarmonicBond") {
                        for b in force
                            .descendants()
                            .filter(|n| n.is_element() && n.has_tag_name("Bond"))
                        {
                            if let (Some(i), Some(j), Some(r0), Some(k)) = (
                                xml_attr_f(b, &["particle1", "p1"]).map(|v| v as usize),
                                xml_attr_f(b, &["particle2", "p2"]).map(|v| v as usize),
                                xml_attr_f(b, &["length", "d"]),
                                xml_attr_f(b, &["k"]),
                            ) {
                                if i < sys.n_particles && j < sys.n_particles {
                                    sys.bonds.push((i, j, r0, k));
                                }
                            }
                        }
                    } else if ftype.contains("HarmonicAngle") {
                        for a in force
                            .descendants()
                            .filter(|n| n.is_element() && n.has_tag_name("Angle"))
                        {
                            if let (Some(i), Some(j), Some(k), Some(t), Some(kk)) = (
                                xml_attr_f(a, &["particle1", "p1"]).map(|v| v as usize),
                                xml_attr_f(a, &["particle2", "p2"]).map(|v| v as usize),
                                xml_attr_f(a, &["particle3", "p3"]).map(|v| v as usize),
                                xml_attr_f(a, &["angle", "a"]),
                                xml_attr_f(a, &["k"]),
                            ) {
                                if i < sys.n_particles && j < sys.n_particles && k < sys.n_particles {
                                    sys.angles.push((i, j, k, t, kk));
                                }
                            }
                        }
                    } else if ftype.contains("PeriodicTorsion") {
                        for t in force
                            .descendants()
                            .filter(|n| n.is_element() && n.has_tag_name("Torsion"))
                        {
                            if let (Some(i), Some(j), Some(k), Some(l)) = (
                                xml_attr_f(t, &["particle1", "p1"]).map(|v| v as usize),
                                xml_attr_f(t, &["particle2", "p2"]).map(|v| v as usize),
                                xml_attr_f(t, &["particle3", "p3"]).map(|v| v as usize),
                                xml_attr_f(t, &["particle4", "p4"]).map(|v| v as usize),
                            ) {
                                let phase = xml_attr_f(t, &["phase"]).unwrap_or(0.0);
                                let kk = xml_attr_f(t, &["k"]).unwrap_or(0.0);
                                let n = xml_attr_f(t, &["periodicity"]).unwrap_or(0.0) as u32;
                                if i < sys.n_particles
                                    && j < sys.n_particles
                                    && k < sys.n_particles
                                    && l < sys.n_particles
                                {
                                    sys.torsions.push((i, j, k, l, phase.to_radians(), kk, n));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if sys.n_particles == 0 {
        return Err(anyhow::anyhow!("system xml '{}': no particles parsed", path));
    }
    if sys.charges.len() != sys.n_particles {
        return Err(anyhow::anyhow!(
            "system xml '{}': NonbondedForce has {} particles but System has {}",
            path,
            sys.charges.len(),
            sys.n_particles
        ));
    }
    Ok(sys)
}

const TIP3P_Q_O: f64 = -0.834;
const TIP3P_Q_H: f64 = 0.417;
const TIP3P_R_OH_NM: f64 = 0.09572;
const TIP3P_K_OH_KJMOL_NM2: f64 = 502416.0;
const TIP3P_THETA_HOH_DEG: f64 = 104.52;
const TIP3P_K_HOH_KJMOL_RAD2: f64 = 627.6;
const TIP3P_SIGMA_O_NM: f64 = 0.315061;
const TIP3P_EPS_O_KJMOL: f64 = 0.635976;
const MASS_O_AMU: f64 = 15.9994;
const MASS_H_AMU: f64 = 1.008;

const WATER_RESNAMES: [&str; 5] = ["SOL", "HOH", "WAT", "TIP3", "TIP"];

#[derive(Debug, Clone, Default)]
pub struct MmTopology {
    pub n_mm: usize,
    pub symbols: Vec<String>,
    pub resnames: Vec<String>,

    pub charges: Vec<f64>,

    pub lj: Vec<(f64, f64)>,

    pub masses: Vec<f64>,

    pub bonds: Vec<(usize, usize, f64, f64)>,

    pub angles: Vec<(usize, usize, usize, f64, f64)>,

    pub box_nm: [f64; 3],

    pub pos_bohr: Vec<f64>,
}

pub fn build_solute_water_box(
    symbols: &[String],
    pos_bohr: &[f64],
    margin_ang: f64,
    spacing_ang: f64,
    max_waters: usize,
) -> (String, usize) {
    const BOHR_ANG: f64 = crate::constants::BOHR;
    let n = symbols.len();
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for a in pos_bohr.chunks(3) {
        for c in 0..3 {
            let v = a[c] * BOHR_ANG;
            lo[c] = lo[c].min(v);
            hi[c] = hi[c].max(v);
        }
    }
    let side = [
        (hi[0] - lo[0]) + 2.0 * margin_ang,
        (hi[1] - lo[1]) + 2.0 * margin_ang,
        (hi[2] - lo[2]) + 2.0 * margin_ang,
    ];
    let center = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let origin = [center[0] - side[0] * 0.5, center[1] - side[1] * 0.5, center[2] - side[2] * 0.5];

    let th = 104.52f64.to_radians();
    let r = 0.9572;
    let h1 = [r * (th / 2.0).sin(), r * (th / 2.0).cos(), 0.0];
    let h2 = [-r * (th / 2.0).sin(), r * (th / 2.0).cos(), 0.0];

    let rotations: Vec<[[f64; 3]; 3]> = (0..24)
        .map(|k| {
            let a = 0.7 * k as f64;
            let b = 1.3 * (k as f64) * 0.618034;
            let c = 2.1 * (k as f64) * 0.415827;
            let (ca, sa, cb, sb, cc, sc) =
                (a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin());
            let rx = [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]];
            let ry = [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]];
            let rz = [[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]];
            fn mul(p: [[f64; 3]; 3], q: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
                let mut r = [[0.0; 3]; 3];
                for i in 0..3 {
                    for jj in 0..3 {
                        r[i][jj] = (0..3).map(|kk| p[i][kk] * q[kk][jj]).sum();
                    }
                }
                r
            }
            mul(rz, mul(ry, rx))
        })
        .collect();

    let clash = |x: f64, y: f64, z: f64, cutoff: f64| -> bool {
        for a in pos_bohr.chunks(3) {
            let dx = a[0] * BOHR_ANG - x;
            let dy = a[1] * BOHR_ANG - y;
            let dz = a[2] * BOHR_ANG - z;
            if (dx * dx + dy * dy + dz * dz).sqrt() < cutoff {
                return true;
            }
        }
        false
    };

    let mut waters: Vec<[f64; 9]> = Vec::new();
    let spacing = spacing_ang.max(2.0);
    let mut gi = 0usize;
    let mut gz = origin[2] + spacing * 0.5;
    while gz < origin[2] + side[2] - spacing * 0.25 {
        let mut gy = origin[1] + spacing * 0.5;
        while gy < origin[1] + side[1] - spacing * 0.25 {
            let mut gx = origin[0] + spacing * 0.5;
            while gx < origin[0] + side[0] - spacing * 0.25 {
                if max_waters == 0 || waters.len() < max_waters {
                    let rot = &rotations[gi % rotations.len()];
                    let o = [gx, gy, gz];
                    let h1 = [o[0] + rot[0][0]*h1[0] + rot[0][1]*h1[1] + rot[0][2]*h1[2],
                              o[1] + rot[1][0]*h1[0] + rot[1][1]*h1[1] + rot[1][2]*h1[2],
                              o[2] + rot[2][0]*h1[0] + rot[2][1]*h1[1] + rot[2][2]*h1[2]];
                    let h2 = [o[0] + rot[0][0]*h2[0] + rot[0][1]*h2[1] + rot[0][2]*h2[2],
                              o[1] + rot[1][0]*h2[0] + rot[1][1]*h2[1] + rot[1][2]*h2[2],
                              o[2] + rot[2][0]*h2[0] + rot[2][1]*h2[1] + rot[2][2]*h2[2]];
                    if !clash(o[0], o[1], o[2], 3.0)
                        && !clash(h1[0], h1[1], h1[2], 2.0)
                        && !clash(h2[0], h2[1], h2[2], 2.0)
                    {
                        waters.push([o[0], o[1], o[2], h1[0], h1[1], h1[2], h2[0], h2[1], h2[2]]);
                    }
                }
                gi += 1;
                gx += spacing;
            }
            gy += spacing;
        }
        gz += spacing;
    }

    let mut out = String::new();
    out.push_str("REST auto-built TIP3P water box\n");
    out.push_str(&format!("{:>5}\n", waters.len() * 3));
    for (w, atoms) in waters.iter().enumerate() {
        let names = ["OW", "HW1", "HW2"];
        for (k, (nm, xyz)) in names.iter().zip(atoms.chunks(3)).enumerate() {

            out.push_str(&format!(
                "{:>5}{:<5}{:>5}{:>5}{:8.3}{:8.3}{:8.3}\n",
                w + 1,
                "SOL",
                nm,
                w * 3 + k + 1,
                xyz[0] / 10.0,
                xyz[1] / 10.0,
                xyz[2] / 10.0
            ));
        }
    }
    out.push_str(&format!("{:10.4}{:10.4}{:10.4}\n", side[0] / 10.0, side[1] / 10.0, side[2] / 10.0));
    (out, waters.len())
}

pub fn parse_gro(path: &str) -> anyhow::Result<MmTopology> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read mm gro file '{}': {}", path, e))?;
    let mut lines = content.lines();
    let _title = lines.next().ok_or_else(|| anyhow::anyhow!("gro file '{}': empty", path))?;
    let natoms_line = lines.next().ok_or_else(|| anyhow::anyhow!("gro file '{}': missing atom count", path))?;
    let natoms: usize = natoms_line
        .trim()
        .parse()
        .map_err(|e| anyhow::anyhow!("gro file '{}': bad atom count '{}': {}", path, natoms_line, e))?;

    let mut top = MmTopology::default();
    let mut unknown_residues: Vec<String> = Vec::new();

    let mut cur_resid: Option<i64> = None;
    let mut water_atoms: Vec<(usize, bool)> = Vec::new();

    let field = |line: &str, a: usize, b: usize| -> String {
        line.get(a..b).unwrap_or("").trim().to_string()
    };

    for (idx, line) in lines.by_ref().take(natoms).enumerate() {
        if line.len() < 44 {
            return Err(anyhow::anyhow!("gro file '{}': atom line {} too short: {:?}", path, idx + 1, line));
        }

        let resid: i64 = field(line, 0, 5).parse().unwrap_or(-1);
        let resname = field(line, 5, 10);
        let atomname = field(line, 10, 15);
        let x: f64 = field(line, 20, 28).parse().map_err(|e| anyhow::anyhow!("gro x '{}': {}", field(line, 20, 28), e))?;
        let y: f64 = field(line, 28, 36).parse().map_err(|e| anyhow::anyhow!("gro y: {}", e))?;
        let z: f64 = field(line, 36, 44).parse().map_err(|e| anyhow::anyhow!("gro z: {}", e))?;

        let symbol = element_from_atomname(&atomname);
        let is_water = WATER_RESNAMES.contains(&resname.as_str());
        let mass = match symbol.as_str() {
            "O" => MASS_O_AMU,
            "H" => MASS_H_AMU,
            _ => crate::geom_io::get_mass_charge(&vec![symbol.clone()])[0].0,
        };
        let charge = if is_water {
            if symbol.eq("O") { TIP3P_Q_O } else { TIP3P_Q_H }
        } else {
            0.0
        };
        let lj = if is_water && symbol.eq("O") {
            (TIP3P_SIGMA_O_NM, TIP3P_EPS_O_KJMOL)
        } else {
            (0.0, 0.0)
        };

        if cur_resid != Some(resid) {
            flush_water(&mut top, &water_atoms);
            water_atoms.clear();
            cur_resid = Some(resid);
            if !is_water && !unknown_residues.iter().any(|r| r.eq(&resname)) {
                unknown_residues.push(resname.clone());
            }
        }
        if is_water {
            water_atoms.push((top.n_mm, symbol.eq("O")));
        }

        top.symbols.push(symbol);
        top.resnames.push(resname);
        top.charges.push(charge);
        top.lj.push(lj);
        top.masses.push(mass);

        top.pos_bohr.push(x * 10.0 / BOHR);
        top.pos_bohr.push(y * 10.0 / BOHR);
        top.pos_bohr.push(z * 10.0 / BOHR);
        top.n_mm += 1;
    }
    flush_water(&mut top, &water_atoms);
    if !unknown_residues.is_empty() {
        eprintln!(
            "Warning: non-water MM residue(s) {:?} have no assigned force field; treated as charge-free LJ-free masses",
            unknown_residues
        );
    }

    let broken: Vec<usize> = top
        .bonds
        .iter()
        .filter(|(i, j, _, _)| {
            let dx = top.pos_bohr[3 * i] - top.pos_bohr[3 * j];
            let dy = top.pos_bohr[3 * i + 1] - top.pos_bohr[3 * j + 1];
            let dz = top.pos_bohr[3 * i + 2] - top.pos_bohr[3 * j + 2];

            (dx * dx + dy * dy + dz * dz).sqrt() > 0.13 * 10.0 / BOHR
        })
        .map(|(i, j, _, _)| *i)
        .collect();
    if !broken.is_empty() {
        eprintln!(
            "Warning: {} O-H bond(s) longer than 0.13 nm in direct coordinates — molecule(s) \
             broken across the periodic boundary. The harmonic bond term will dump huge strain \
             into the first force evaluation. Fix the input with whole-molecule treatment \
             (e.g. `gmx trjconv -pbc whole`) before running REST MD.",
            broken.len()
        );
    }

    let box_line = lines.next().unwrap_or("");
    let box_nums: Vec<f64> = box_line
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    if box_nums.len() >= 3 {
        top.box_nm = [box_nums[0], box_nums[1], box_nums[2]];
    }
    Ok(top)
}

fn flush_water(top: &mut MmTopology, water_atoms: &[(usize, bool)]) {
    let oxygens: Vec<usize> = water_atoms.iter().filter(|(_, o)| *o).map(|(i, _)| *i).collect();
    let hydrogens: Vec<usize> = water_atoms.iter().filter(|(_, o)| !*o).map(|(i, _)| *i).collect();
    if oxygens.len() != 1 || hydrogens.len() != 2 {
        if !water_atoms.is_empty() {
            eprintln!(
                "Warning: skipped malformed water residue with {} O and {} H atoms",
                oxygens.len(),
                hydrogens.len()
            );
        }
        return;
    }
    let o = oxygens[0];
    for h in hydrogens.iter() {
        top.bonds.push((o, *h, TIP3P_R_OH_NM, TIP3P_K_OH_KJMOL_NM2));
    }
    let (h1, h2) = (hydrogens[0], hydrogens[1]);
    top.angles.push((
        h1,
        o,
        h2,
        TIP3P_THETA_HOH_DEG.to_radians(),
        TIP3P_K_HOH_KJMOL_RAD2,
    ));
}

fn element_from_atomname(name: &str) -> String {
    let letters: String = name.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    if letters.len() >= 2 {
        let mut chars = letters.chars();
        let first = chars.next().unwrap();
        let second = chars.next().unwrap();
        if second.is_lowercase() {
            return format!("{}{}", first, second);
        }
    }
    letters.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_else(|| "X".to_string())
}

pub struct OpenMmEngine {
    driver: Py<PyAny>,
}

const PY_HELPER: &str = r#"
class _RestOpenMM:
    def __init__(self, masses, charges, sigma, epsilon, bonds, angles, torsions, exceptions, box_nm, pbc):
        import openmm as mm
        print('OpenMM version: %s' % mm.version.version, flush=True)
        import openmm.unit as u
        self._u = u
        sys = mm.System()
        self._sys = sys
        for m in masses:
            sys.addParticle(float(m))
        if pbc:
            sys.setDefaultPeriodicBoxVectors(
                mm.Vec3(float(box_nm[0]), 0.0, 0.0),
                mm.Vec3(0.0, float(box_nm[1]), 0.0),
                mm.Vec3(0.0, 0.0, float(box_nm[2])))
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.PME if pbc else mm.NonbondedForce.NoCutoff)
        if pbc:
            half_box = 0.5 * min(float(box_nm[0]), float(box_nm[1]), float(box_nm[2]))
            nb.setCutoffDistance(min(1.0, 0.99 * half_box))
        for q, s, e in zip(charges, sigma, epsilon):
            nb.addParticle(float(q), float(s), float(e))
        for (i, j) in exceptions:
            nb.addException(int(i), int(j), 0.0, 0.0, 0.0)
        sys.addForce(nb)
        if bonds:
            bf = mm.HarmonicBondForce()
            for (i, j, r0, k) in bonds:
                bf.addBond(int(i), int(j), float(r0), float(k))
            sys.addForce(bf)
        if angles:
            af = mm.HarmonicAngleForce()
            for (i, j, k, th0, kth) in angles:
                af.addAngle(int(i), int(j), int(k), float(th0), float(kth))
            sys.addForce(af)
        if torsions:
            tf = mm.PeriodicTorsionForce()
            for (i, j, k, l, phase, kk, n) in torsions:
                tf.addTorsion(int(i), int(j), int(k), int(l), int(n), float(phase), float(kk))
            sys.addForce(tf)
        integ = mm.VerletIntegrator(0.001)
        platform = mm.Platform.getPlatformByName('CPU')
        self.context = mm.Context(sys, integ, platform)

    def evaluate(self, pos_nm_flat):
        u = self._u
        n = len(pos_nm_flat) // 3
        expected = self._sys.getNumParticles()
        if n != expected:
            raise ValueError(
                f'REST MD: {n} positions passed to an OpenMM context with '
                f'{expected} particles - coordinate/force-field layout mismatch '
                f'(check qmmm_gro_skip and the GRO/[geom] agreement)')
        pos = u.Quantity([tuple(pos_nm_flat[3 * i:3 * i + 3]) for i in range(n)], u.nanometer)
        self.context.setPositions(pos)
        st = self.context.getState(getEnergy=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        if abs(e) > 1e7:
            raise ValueError(
                f'REST MD: OpenMM E_MM = {e:.3e} kJ/mol is unphysical '
                f'(~{e / 96.485:.2e} eV) - overlapping particles or a force-field '
                f'layout mismatch; refusing to propagate')
        f = st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer)
        return e, [c for v in f for c in v]
"#;

const PY_XML_HELPER: &str = r#"
class _RestOpenMMXml:
    def __init__(self, xml_path):
        import openmm as mm
        print('OpenMM version: %s' % mm.version.version, flush=True)
        import openmm.unit as u
        self._u = u
        with open(xml_path) as fh:
            sys = mm.XmlSerializer.deserialize(fh.read())
        self._sys = sys
        integ = mm.VerletIntegrator(0.001)
        platform = mm.Platform.getPlatformByName('CPU')
        self.context = mm.Context(sys, integ, platform)

    def evaluate(self, pos_nm_flat):
        u = self._u
        n = len(pos_nm_flat) // 3
        expected = self._sys.getNumParticles()
        if n != expected:
            raise ValueError(
                f'REST MD: {n} positions passed to an OpenMM context with '
                f'{expected} particles - coordinate/force-field layout mismatch '
                f'(check qmmm_gro_skip and the GRO/[geom] agreement)')
        pos = u.Quantity([tuple(pos_nm_flat[3 * i:3 * i + 3]) for i in range(n)], u.nanometer)
        self.context.setPositions(pos)
        st = self.context.getState(getEnergy=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        if abs(e) > 1e7:
            raise ValueError(
                f'REST MD: OpenMM E_MM = {e:.3e} kJ/mol is unphysical '
                f'(~{e / 96.485:.2e} eV) - overlapping particles or a force-field '
                f'layout mismatch; refusing to propagate')
        f = st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer)
        return e, [c for v in f for c in v]
"#;

impl OpenMmEngine {
    fn build_engine(
        masses: Vec<f64>,
        charges: Vec<f64>,
        sigma_nm: Vec<f64>,
        eps_kjmol: Vec<f64>,
        bonds: Vec<(usize, usize, f64, f64)>,
        angles: Vec<(usize, usize, usize, f64, f64)>,
        torsions: Vec<(usize, usize, usize, usize, f64, f64, u32)>,
        exceptions: Vec<(usize, usize)>,
        box_vec: Vec<f64>,
        pbc: bool,
    ) -> anyhow::Result<Self> {
        pyo3::prepare_freethreaded_python();
        let driver = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let code = std::ffi::CString::new(PY_HELPER).unwrap();
            let locals = PyDict::new(py);
            py.run(&code, None, Some(&locals))?;
            let cls = locals
                .get_item("_RestOpenMM")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("_RestOpenMM missing"))?;
            let obj = cls.call1((
                masses,
                charges,
                sigma_nm,
                eps_kjmol,
                bonds,
                angles,
                torsions,
                exceptions,
                box_vec,
                pbc,
            ))?;
            Ok(obj.into())
        })
        .map_err(|e| anyhow::anyhow!("OpenMM engine construction failed: {}", e))?;
        Ok(OpenMmEngine { driver })
    }

    pub fn new(
        qm_symbols: &[String],
        qm_masses_amu: &[f64],
        qm_lj_table: &std::collections::BTreeMap<String, [f64; 2]>,
        mm: &MmTopology,
        pbc: bool,
        gro_skip: usize,
    ) -> anyhow::Result<Self> {
        let n_total = qm_symbols.len() + mm.n_mm;

        let mut masses: Vec<f64> = Vec::with_capacity(n_total);
        let mut charges: Vec<f64> = Vec::with_capacity(n_total);
        let mut sigma_nm: Vec<f64> = Vec::with_capacity(n_total);
        let mut eps_kjmol: Vec<f64> = Vec::with_capacity(n_total);

        for (sym, m) in qm_symbols.iter().zip(qm_masses_amu.iter()) {
            masses.push(*m);
            charges.push(0.0);
            match qm_lj_table.get(sym) {
                Some([s_ang, e_kj]) => {
                    sigma_nm.push(*s_ang / 10.0);
                    eps_kjmol.push(*e_kj);
                }
                _ => {
                    eprintln!("Warning: no QM-MM LJ parameters for element {}; using none", sym);
                    sigma_nm.push(0.0);
                    eps_kjmol.push(0.0);
                }
            }
        }
        for i in 0..mm.n_mm {
            masses.push(mm.masses[i]);
            charges.push(mm.charges[i]);
            sigma_nm.push(mm.lj[i].0);
            eps_kjmol.push(mm.lj[i].1);
        }

        let n_qm = qm_symbols.len();
        let bonds: Vec<(usize, usize, f64, f64)> = mm
            .bonds
            .iter()
            .map(|(i, j, r0, k)| (i + n_qm, j + n_qm, *r0, *k))
            .collect();
        let angles: Vec<(usize, usize, usize, f64, f64)> = mm
            .angles
            .iter()
            .map(|(i, j, k, t0, kt)| (i + n_qm, j + n_qm, k + n_qm, *t0, *kt))
            .collect();

        let mut exceptions: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_qm {
            for j in (i + 1)..n_qm {
                exceptions.push((i, j));
            }
        }

        for i in 0..n_qm {
            for j in 0..gro_skip.min(n_qm) {
                exceptions.push((i.min(n_qm + j), i.max(n_qm + j)));
            }
        }
        for (i, j, _, _) in mm.bonds.iter() {
            exceptions.push((i + n_qm, j + n_qm));
        }

        for (i, _j, k, _, _) in mm.angles.iter() {
            exceptions.push((i + n_qm, k + n_qm));
        }

        let box_vec: Vec<f64> = mm.box_nm.to_vec();
        Self::build_engine(
            masses, charges, sigma_nm, eps_kjmol, bonds, angles, Vec::new(), exceptions,
            box_vec, pbc,
        )
    }

    pub fn new_from_system(
        sys: &MmSystem,
        box_nm: [f64; 3],
        pbc: bool,
    ) -> anyhow::Result<Self> {
        let box_vec: Vec<f64> = if sys.box_nm[0] > 0.0 { sys.box_nm.to_vec() } else { box_nm.to_vec() };
        Self::build_engine(
            sys.masses.clone(),
            sys.charges.clone(),
            sys.sigma_nm.clone(),
            sys.eps_kjmol.clone(),
            sys.bonds.clone(),
            sys.angles.clone(),
            sys.torsions.clone(),
            sys.exceptions.clone(),
            box_vec,
            pbc,
        )
    }

    pub fn new_from_system_xml(xml_path: &str) -> anyhow::Result<Self> {
        pyo3::prepare_freethreaded_python();
        let driver = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let code = std::ffi::CString::new(PY_XML_HELPER).unwrap();
            let locals = PyDict::new(py);
            py.run(&code, None, Some(&locals))?;
            let cls = locals
                .get_item("_RestOpenMMXml")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("_RestOpenMMXml missing"))?;
            let obj = cls.call1((xml_path,))?;
            Ok(obj.into())
        })
        .map_err(|e| anyhow::anyhow!("OpenMM XML engine construction failed: {}", e))?;
        Ok(OpenMmEngine { driver })
    }

    pub fn evaluate(&self, pos_bohr: &[f64]) -> anyhow::Result<(f64, Vec<f64>)> {
        let pos_nm: Vec<f64> = pos_bohr.iter().map(|x| x * BOHR / 10.0).collect();
        let (e_kjmol, f_kjmol_nm) = Python::with_gil(|py| -> PyResult<(f64, Vec<f64>)> {
            let res = self.driver.bind(py).call_method1("evaluate", (pos_nm,))?;

            let e: f64 = res.get_item(0)?.extract()?;
            let f: Vec<f64> = res.get_item(1)?.extract()?;
            Ok((e, f))
        })
        .map_err(|e| anyhow::anyhow!("OpenMM evaluation failed: {}", e))?;
        let e_ha = e_kjmol / HARTREE2KJMOL;

        let f_conv = 1.0 / (EV2KJMOL * 10.0 * AU_FORCE2_EV_PER_ANG);
        let f_au: Vec<f64> = f_kjmol_nm.iter().map(|x| x * f_conv).collect();
        Ok((e_ha, f_au))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingSite {

    pub mm_index: usize,
    pub charge: f64,

    pub pos_bohr: [f64; 3],
}

pub fn build_embedding(
    qm_pos_bohr: &[f64],
    mm_pos_bohr: &[f64],
    mm_charges: &[f64],
    box_nm: [f64; 3],
    pbc: bool,
    cutoff_ang: f64,
) -> Vec<EmbeddingSite> {
    let n_qm = qm_pos_bohr.len() / 3;
    let n_mm = mm_pos_bohr.len() / 3;
    let box_bohr: [f64; 3] = [
        box_nm[0] * 10.0 / BOHR,
        box_nm[1] * 10.0 / BOHR,
        box_nm[2] * 10.0 / BOHR,
    ];
    let cutoff_bohr = cutoff_ang / BOHR;
    let mut sites = Vec::new();
    for j in 0..n_mm {
        if mm_charges[j] == 0.0 {
            continue;
        }
        let mut best_dist2 = f64::INFINITY;
        let mut best_pos = [0.0f64; 3];
        for i in 0..n_qm {
            let mut d = [
                mm_pos_bohr[3 * j] - qm_pos_bohr[3 * i],
                mm_pos_bohr[3 * j + 1] - qm_pos_bohr[3 * i + 1],
                mm_pos_bohr[3 * j + 2] - qm_pos_bohr[3 * i + 2],
            ];
            if pbc {
                for c in 0..3 {
                    d[c] -= box_bohr[c] * (d[c] / box_bohr[c]).round();
                }
            }
            let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if dist2 < best_dist2 {
                best_dist2 = dist2;
                best_pos = [
                    qm_pos_bohr[3 * i] + d[0],
                    qm_pos_bohr[3 * i + 1] + d[1],
                    qm_pos_bohr[3 * i + 2] + d[2],
                ];
            }
        }
        if best_dist2 <= cutoff_bohr * cutoff_bohr {
            sites.push(EmbeddingSite {
                mm_index: j,
                charge: mm_charges[j],
                pos_bohr: best_pos,
            });
        }
    }
    sites
}

const PY_FF_BUILDER: &str = r#"
def _permute_system(sys, order):
    import openmm as mm
    n = sys.getNumParticles()
    inv = [0] * n
    for nw, old in enumerate(order):
        inv[old] = nw
    out = mm.System()
    for nw in range(n):
        out.addParticle(sys.getParticleMass(order[nw]))
    try:
        bv = sys.getDefaultPeriodicBoxVectors()
        out.setDefaultPeriodicBoxVectors(bv[0], bv[1], bv[2])
    except Exception:
        pass
    for fi in range(sys.getNumForces()):
        f = sys.getForce(fi)
        cn = f.__class__.__name__
        nf = None
        if cn == 'NonbondedForce':
            nf = mm.NonbondedForce()
            nf.setNonbondedMethod(f.getNonbondedMethod())
            nf.setCutoffDistance(f.getCutoffDistance())
            nf.setEwaldErrorTolerance(f.getEwaldErrorTolerance())
            try:
                nf.setUseDispersionCorrection(f.getUseDispersionCorrection())
            except Exception:
                pass
            for nw in range(n):
                q, s, e = f.getParticleParameters(order[nw])
                nf.addParticle(q, s, e)
            for k in range(f.getNumExceptions()):
                a, b, q, s, e = f.getExceptionParameters(k)
                nf.addException(inv[a], inv[b], q, s, e)
        elif cn == 'HarmonicBondForce':
            nf = mm.HarmonicBondForce()
            for k in range(f.getNumBonds()):
                a, b, r, kk = f.getBondParameters(k)
                nf.addBond(inv[a], inv[b], r, kk)
        elif cn == 'HarmonicAngleForce':
            nf = mm.HarmonicAngleForce()
            for k in range(f.getNumAngles()):
                a, b, c, th, kk = f.getAngleParameters(k)
                nf.addAngle(inv[a], inv[b], inv[c], th, kk)
        elif cn == 'PeriodicTorsionForce':
            nf = mm.PeriodicTorsionForce()
            for k in range(f.getNumTorsions()):
                a, b, c, d, per, ph, kk = f.getTorsionParameters(k)
                nf.addTorsion(inv[a], inv[b], inv[c], inv[d], per, ph, kk)
        elif cn == 'CustomTorsionForce':
            nf = mm.CustomTorsionForce(f.getEnergyFunction())
            for p in range(f.getNumPerTorsionParameters()):
                nf.addPerTorsionParameter(f.getPerTorsionParameterName(p))
            for k in range(f.getNumTorsions()):
                a, b, c, d, params = f.getTorsionParameters(k)
                nf.addTorsion(inv[a], inv[b], inv[c], inv[d], params)
        elif cn == 'RBTorsionForce':
            nf = mm.RBTorsionForce()
            for k in range(f.getNumTorsions()):
                a, b, c, d, c0, c1, c2, c3, c4, c5 = f.getTorsionParameters(k)
                nf.addTorsion(inv[a], inv[b], inv[c], inv[d], c0, c1, c2, c3, c4, c5)
        elif cn == 'CMAPTorsionForce':
            nf = mm.CMAPTorsionForce()
            for mp in range(f.getNumMaps()):
                size, energy = f.getMapParameters(mp)
                nf.addMap(size, list(energy))
            for k in range(f.getNumTorsions()):
                a, b, c, d, e1, f1 = f.getTorsionParameters(k)[:6]
                nf.addTorsion(a, inv[b], inv[c], inv[d], inv[e1], inv[f1])
        else:
            print('WARNING: force %s not permuted (skipped)' % cn)
        if nf is not None:
            nf.setForceGroup(f.getForceGroup())
            out.addForce(nf)
    return out


def _rest_build_system(pdb_path, ff_files, qm_atoms, qm_skip, links, frontier,
                       rescale_mode, pbc, out_xml, out_gro):
    import openmm as mm
    print('OpenMM version: %s' % mm.version.version, flush=True)
    import openmm.app as app
    import openmm.unit as u
    from openmm.app import PDBFile, ForceField, Modeller
    pdf = PDBFile(pdb_path)
    ff = ForceField(*ff_files)
    mod = Modeller(pdf.topology, pdf.positions)
    top = mod.topology
    box = top.getPeriodicBoxVectors()
    has_box = box is not None
    if pbc and not has_box:
        raise ValueError('qmmm_pbc = true but the input PDB has no CRYST1/periodic box')
    method = app.PME if (pbc and has_box) else app.NoCutoff
    sys = ff.createSystem(
        top, nonbondedMethod=method, nonbondedCutoff=1.0*u.nanometer,
        constraints=None, rigidWater=False, removeCMMotion=False)
    n = sys.getNumParticles()
    qm = set(int(i) for i in qm_atoms) if len(qm_atoms) > 0 else set(range(min(qm_skip, n)))
    for i in qm:
        if i < 0 or i >= n:
            raise ValueError('qmmm_qm_atoms index %d out of range' % i)
    fr = set(int(x) for x in frontier)
    for (qb, mh) in links:
        fr.add(int(mh))
    fr -= qm
    nb = None
    for f in sys.getForces():
        if f.__class__.__name__ == 'NonbondedForce':
            nb = f
    toms = list(top.atoms())
    chem = [(a.element.symbol if a.element is not None else 'X') for a in toms]
    reskey = [(a.residue.name, a.residue.id) for a in toms]
    by_res = {}
    for i in range(n):
        by_res.setdefault(reskey[i], []).append(i)
    if nb is not None:
        final = [nb.getParticleParameters(i)[0]._value for i in range(n)]
        if fr:
            rem = {k: 0.0 for k in by_res}
            for i in fr:
                rem[reskey[i]] += final[i]
                final[i] = 0.0
            if rescale_mode == 'residue':
                for k, rest in by_res.items():
                    rest = [i for i in rest if i not in fr and i not in qm]
                    if rest and abs(rem[k]) > 1.0e-12:
                        for i in rest:
                            final[i] += rem[k] / len(rest)
        for i in sorted(qm):
            final[i] = 0.0
        for i in range(n):
            q, s, e = nb.getParticleParameters(i)
            nb.setParticleParameters(i, final[i], s, e)
        ex_idx = {}
        for k in range(nb.getNumExceptions()):
            a, b, q, s, e = nb.getExceptionParameters(k)
            ex_idx[(min(a, b), max(a, b))] = k
            if a in qm and b in qm:
                nb.setExceptionParameters(k, a, b, 0.0, 1.0, 0.0)
        ql = sorted(qm)
        for i in range(len(ql)):
            for j in range(i + 1, len(ql)):
                key = (ql[i], ql[j])
                if key not in ex_idx:
                    nb.addException(key[0], key[1], 0.0, 1.0, 0.0)
    for fi in range(sys.getNumForces()):
        f = sys.getForce(fi)
        cn = f.__class__.__name__
        if cn == 'HarmonicBondForce':
            for k in range(f.getNumBonds()):
                a, b, r, kk = f.getBondParameters(k)
                if a in qm and b in qm:
                    f.setBondParameters(k, a, b, r, 0.0)
        elif cn == 'HarmonicAngleForce':
            for k in range(f.getNumAngles()):
                a, b, c, th, kk = f.getAngleParameters(k)
                if a in qm and b in qm and c in qm:
                    f.setAngleParameters(k, a, b, c, th, 0.0)
        elif cn == 'PeriodicTorsionForce':
            for k in range(f.getNumTorsions()):
                a, b, c, d, per, ph, kk = f.getTorsionParameters(k)
                if a in qm and b in qm and c in qm and d in qm:
                    f.setTorsionParameters(k, a, b, c, d, per, ph, 0.0)
    order = sorted(qm) + [i for i in range(n) if i not in qm]
    sys2 = _permute_system(sys, order)
    with open(out_xml, 'w') as fh:
        fh.write(mm.XmlSerializer.serialize(sys2))
    pos = mod.positions.value_in_unit(u.nanometer)
    lines = []
    for nw, old in enumerate(order):
        res = toms[old].residue
        lines.append('%5d%-5s%5s%5d%8.3f%8.3f%8.3f' % (
            res.index + 1, res.name[:5], chem[old][:5], nw + 1,
            pos[old][0], pos[old][1], pos[old][2]))
    boxline = '%10.5f%10.5f%10.5f' % (
        box[0][0] / u.nanometer if has_box else 0.0,
        box[1][1] / u.nanometer if has_box else 0.0,
        box[2][2] / u.nanometer if has_box else 0.0)
    with open(out_gro, 'w') as fh:
        fh.write('REST auto-built from PDB via OpenMM ForceField (QM atoms first)\n')
        fh.write('%5d\n' % n)
        fh.write('\n'.join(lines) + '\n')
        fh.write(boxline + '\n')
    import os
    qm_xyz = os.path.join(os.path.dirname(out_xml) or '.', 'auto_qm.xyz')
    with open(qm_xyz, 'w') as fh:
        fh.write('%d\n' % len(qm))
        fh.write('REST auto QM region (append link H after these, in order)\n')
        for nw in range(len(qm)):
            old = order[nw]
            r = pos[old]
            fh.write('%-2s %15.8f %15.8f %15.8f\n' % (chem[old], r[0], r[1], r[2]))
    return n
"#;

pub fn build_system_from_pdb(
    pdb_path: &str,
    ff_files: &[String],
    qm_atoms: &[usize],
    qm_skip: usize,
    links: &[(usize, usize)],
    frontier: &[usize],
    rescale_mode: &str,
    pbc: bool,
    out_xml: &str,
    out_gro: &str,
) -> anyhow::Result<usize> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<usize> {
        let code = std::ffi::CString::new(PY_FF_BUILDER).unwrap();
        let locals = PyDict::new(py);
        py.run(&code, Some(&locals), Some(&locals))?;
        let func = locals
            .get_item("_rest_build_system")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("_rest_build_system missing"))?;
        let n: usize = func
            .call1((
                pdb_path,
                ff_files.to_vec(),
                qm_atoms.to_vec(),
                qm_skip,
                links.to_vec(),
                frontier.to_vec(),
                rescale_mode,
                pbc,
                out_xml,
                out_gro,
            ))?
            .extract()?;
        Ok(n)
    })
    .map_err(|e| anyhow::anyhow!("OpenMM force-field system build failed: {}", e))
}
