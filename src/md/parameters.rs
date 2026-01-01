use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdParameters {

    #[serde(default = "default_ensemble")]
    pub ensemble: String,

    #[serde(default = "default_dt")]
    pub dt: f64,

    #[serde(default = "default_steps")]
    pub steps: usize,

    #[serde(default = "default_temperature")]
    pub temperature: f64,

    #[serde(default = "default_friction")]
    pub friction: f64,

    #[serde(default = "default_friction_units")]
    pub friction_units: String,

    #[serde(default)]
    pub seed: u64,

    #[serde(default = "default_init_velocities")]
    pub init_velocities: String,

    #[serde(default)]
    pub velocities_file: String,

    #[serde(default)]
    pub restart_input: String,

    #[serde(default = "one")]
    pub traj_interval: usize,
    pub equil_steps: usize,
    #[serde(default = "default_opt_fmax")]
    pub opt_fmax: f64,
    #[serde(default = "default_opt_steps")]
    pub opt_steps: usize,

    #[serde(default)]
    pub out_prefix: String,

    #[serde(default = "default_restart_output")]
    pub restart_output: String,

    #[serde(default = "default_outputs")]
    pub outputs: Vec<String>,

    #[serde(default)]
    pub qmmm: Option<QmmmParams>,

    #[serde(default)]
    pub umbrella: Option<UmbrellaParams>,
}

pub const OUTPUT_SWITCHES: [&str; 6] =
    ["md_log", "traj", "dipole", "energy_force", "umbrella_csv", "restart"];

fn default_outputs_for(qmmm: bool) -> Vec<String> {
    let mut out = vec![String::from("md_log"), String::from("traj")];
    out.push(String::from(if qmmm { "energy_force" } else { "dipole" }));
    out.push(String::from("restart"));
    out
}

fn default_outputs() -> Vec<String> {
    default_outputs_for(false)
}

impl MdParameters {

    pub fn friction_fs_inv(&self) -> f64 {
        match self.friction_units.as_str() {
            "fs_inv" => self.friction,
            _ => self.friction / super::ase::ASE_TIME_PER_FS,
        }
    }

    pub fn output_enabled(&self, name: &str) -> bool {
        self.outputs.iter().any(|s| s.eq_ignore_ascii_case(name))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QmmmParams {

    #[serde(default)]
    pub mm_file: String,

    #[serde(default = "default_water_model")]
    pub water_model: String,

    #[serde(default = "default_mm_cutoff")]
    pub mm_cutoff: f64,

    #[serde(default = "default_pbc")]
    pub pbc: bool,

    #[serde(default = "default_lj_table")]
    pub lj: BTreeMap<String, [f64; 2]>,

    #[serde(default)]
    pub system_xml: String,

    #[serde(default)]
    pub links: Vec<(usize, usize)>,

    #[serde(default = "default_link_r_eq")]
    pub link_r_eq: f64,

    #[serde(default = "default_nb_cutoff")]
    pub nb_cutoff: f64,

    #[serde(default)]
    pub gro_skip: usize,

    #[serde(default)]
    pub build_box: bool,
    #[serde(default = "default_box_margin")]
    pub box_margin: f64,
    #[serde(default = "default_box_spacing")]
    pub box_spacing: f64,

    #[serde(default)]
    pub build_system: bool,
    #[serde(default)]
    pub ff: Vec<String>,
    #[serde(default)]
    pub mm_pdb: String,
    #[serde(default)]
    pub frontier: Vec<usize>,
    #[serde(default)]
    pub top: String,
    #[serde(default)]
    pub ff_dir: String,
    #[serde(default)]
    pub qm_atoms: Vec<usize>,
    #[serde(default = "default_frontier_rescale")]
    pub frontier_rescale: String,

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmbrellaParams {

    #[serde(default = "default_cv")]
    pub cv: String,

    #[serde(default)]
    pub atoms: Vec<usize>,

    #[serde(default)]
    pub center: f64,

    #[serde(default = "default_kappa")]
    pub kappa: f64,

    #[serde(default = "default_potential")]
    pub potential: String,

    #[serde(default)]
    pub sum_kappa: f64,
    #[serde(default)]
    pub sum_center: f64,
}

fn one() -> usize { 10 }
fn default_ensemble() -> String { String::from("nvt") }
fn default_dt() -> f64 { 0.5 }
fn default_steps() -> usize { 100 }
fn default_temperature() -> f64 { 300.0 }
fn default_friction() -> f64 { 0.02 }
fn default_friction_units() -> String { String::from("ase") }
fn default_opt_fmax() -> f64 { 0.02 }
fn default_opt_steps() -> usize { 200 }
fn default_box_margin() -> f64 { 11.0 }
fn default_box_spacing() -> f64 { 3.1 }
fn default_init_velocities() -> String { String::from("maxwell") }
fn default_restart_output() -> String { String::from("md_restart") }
fn default_water_model() -> String { String::from("tip3p") }
fn default_mm_cutoff() -> f64 { 10.0 }
fn default_pbc() -> bool { true }
fn default_kappa() -> f64 { 400.0 }
fn default_potential() -> String { String::from("cosine") }
fn default_cv() -> String { String::from("dihedral") }
fn default_link_r_eq() -> f64 { 1.09 }
fn default_nb_cutoff() -> f64 { 1.0 }
fn default_frontier_rescale() -> String { String::from("residue") }

fn default_lj_table() -> BTreeMap<String, [f64; 2]> {
    BTreeMap::from([
        (String::from("C"), [3.39967, 0.457730]),
        (String::from("H"), [2.64953, 0.065689]),
        (String::from("O"), [3.15061, 0.635976]),
    ])
}

impl MdParameters {

    pub fn print_interval(&self, step: usize) -> bool {
        let every = (self.steps / 20).max(1);
        step % every == 0 || step == self.steps
    }
}

impl UmbrellaParams {

    pub fn cv_type(&self) -> super::umbrella::CvType {
        match self.cv.to_lowercase().as_str() {
            "distance" => super::umbrella::CvType::Distance,
            "angle" => super::umbrella::CvType::Angle,
            "distance_diff" | "distance difference" => super::umbrella::CvType::DistanceDiff,
            _ => super::umbrella::CvType::Dihedral,
        }
    }

    pub fn umbrella_potential(&self) -> super::umbrella::BiasPotential {
        if self.potential.eq("harmonic") {
            super::umbrella::BiasPotential::Harmonic
        } else {
            super::umbrella::BiasPotential::Cosine
        }
    }
}

impl Default for MdParameters {
    fn default() -> Self {
        MdParameters {
            ensemble: default_ensemble(),
            dt: default_dt(),
            steps: default_steps(),
            opt_fmax: default_opt_fmax(),
            opt_steps: default_opt_steps(),
            temperature: default_temperature(),
            friction: default_friction(),
            friction_units: default_friction_units(),
            seed: 0,
            init_velocities: default_init_velocities(),
            velocities_file: String::new(),
            restart_input: String::new(),
            traj_interval: one(),
            equil_steps: 0,
            out_prefix: String::new(),
            restart_output: default_restart_output(),
            outputs: default_outputs(),
            qmmm: None,
            umbrella: None,
        }
    }
}

impl Default for QmmmParams {
    fn default() -> Self {
        QmmmParams {
            mm_file: String::new(),
            water_model: default_water_model(),
            mm_cutoff: default_mm_cutoff(),
            pbc: default_pbc(),
            lj: default_lj_table(),
            system_xml: String::new(),
            links: Vec::new(),
            link_r_eq: default_link_r_eq(),
            nb_cutoff: default_nb_cutoff(),
            gro_skip: 0,
            build_box: false,
            box_margin: default_box_margin(),
            box_spacing: default_box_spacing(),
            build_system: false,
            ff: Vec::new(),
            mm_pdb: String::new(),
            frontier: Vec::new(),
            top: String::new(),
            ff_dir: String::new(),
            qm_atoms: Vec::new(),
            frontier_rescale: default_frontier_rescale(),
        }
    }
}

impl Default for UmbrellaParams {
    fn default() -> Self {
        UmbrellaParams {
            cv: default_cv(),
            atoms: vec![],
            center: 0.0,
            kappa: default_kappa(),
            potential: default_potential(),
            sum_kappa: 0.0,
            sum_center: 0.0,
        }
    }
}

impl MdParameters {

    pub fn formated_report(&self) -> String {
        let mut out = format!(
            "MD ensemble: {}, dt = {} fs, steps = {}, T = {} K",
            self.ensemble, self.dt, self.steps, self.temperature
        );
        if self.ensemble.eq("nvt") {
            let tau = if self.friction > 0.0 {
                format!("{:.1} fs", 1.0 / self.friction_fs_inv())
            } else {
                "inf".to_string()
            };
            out.push_str(&format!(
                ", friction = {} {} (tau = {})",
                self.friction, self.friction_units, tau
            ));
        }
        out.push_str("\nMD engine: ASE (ase.md Langevin/VelocityVerlet)");
        if let Some(qmmm) = &self.qmmm {
            out.push_str(&format!(
                "\nQM/MM enabled: mm_file = {}, water_model = {}, mm_cutoff = {} A, pbc = {}",
                qmmm.mm_file, qmmm.water_model, qmmm.mm_cutoff, qmmm.pbc
            ));
        }
        if let Some(umb) = &self.umbrella {
            let kappa_unit = match (umb.potential.as_str(), umb.cv.as_str()) {
                ("harmonic", "distance") | ("harmonic", "distance_diff") => "kcal/mol/A^2",
                ("harmonic", _) => "kcal/mol/rad^2",
                _ => "kcal/mol",
            };
            out.push_str(&format!(
                "\nUmbrella bias: potential = {}, atoms = {:?}, center = {} deg, kappa = {} {}",
                umb.potential, umb.atoms, umb.center, umb.kappa, kappa_unit
            ));
            if umb.sum_kappa > 0.0 {
                out.push_str(&format!(
                    ", sum restraint: kappa_sum = {} kcal/mol/A^2, s0 = {} A",
                    umb.sum_kappa, umb.sum_center
                ));
            }
        }
        out.push_str(&format!("\nMD outputs: {}", self.outputs.join(", ")));
        out
    }
}

fn parse_usize_list(value: &serde_json::Value) -> Vec<usize> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_u64().map(|x| x as usize))
            .collect(),
        serde_json::Value::Number(num) => {
            num.as_u64().map(|x| vec![x as usize]).unwrap_or_default()
        }
        serde_json::Value::String(text) => text
            .trim_matches(|c| c == '[' || c == ']')
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .collect(),
        _ => Vec::new(),
    }
}

const MD_KEYS: &[&str] = &[
    "ensemble", "dt", "steps", "temperature", "friction", "friction_units",
    "seed",
    "init_velocities", "velocities_file", "traj_interval", "out_prefix",
    "restart_output", "outputs", "qmmm_mm_file", "qmmm_water_model", "qmmm_mm_cutoff",
    "qmmm_system_xml", "qmmm_links", "qmmm_link_r_eq", "qmmm_gro_skip",
    "equil_steps",
    "qmmm_pbc", "qmmm_lj", "mm_system_xml",
    "qmmm_build_box", "qmmm_box_margin", "qmmm_box_spacing",
    "qmmm_build_system", "qmmm_ff", "qmmm_mm_pdb", "qmmm_frontier",
    "qmmm_top", "qmmm_ff_dir", "qmmm_qm_atoms", "qmmm_frontier_rescale",
    "qmmm_nb_cutoff",
    "opt_fmax", "opt_steps",
    "umbrella_atoms", "umbrella_center",
    "umbrella_kappa", "umbrella_potential", "umbrella_cv",
    "umbrella_sum_kappa", "umbrella_sum_center",
    "restart_input",
];

pub fn parse_md_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<MdParameters>> {
    let md_section = tmp_keys.get("md").filter(|v| v.is_object());
    let ctrl_section = tmp_keys.get("ctrl").filter(|v| v.is_object());

    let ctrl_has_md_keys = ctrl_section.map_or(false, |c| {
        MD_KEYS.iter().any(|k| c.get(*k).is_some())
    });
    if md_section.is_none() && !ctrl_has_md_keys {
        return Ok(None);
    }

    let mut merged = serde_json::Map::new();
    if let Some(serde_json::Value::Object(c)) = ctrl_section {
        for k in MD_KEYS.iter() {
            if let Some(v) = c.get(*k) {
                merged.insert(k.to_string(), v.clone());
            }
        }
    }
    if let Some(serde_json::Value::Object(m)) = md_section {
        for (k, v) in m.iter() {
            merged.insert(k.clone(), v.clone());
        }
    }

    let get_num = |key: &str| merged.get(key)
        .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok())));
    let get_str = |key: &str| merged.get(key).and_then(|v| v.as_str()).map(|s| s.to_string());
    let get_u64 = |key: &str| merged.get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok())));
    let get_bool = |key: &str| merged.get(key)
        .and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s.eq_ignore_ascii_case("true"))));

    let mut p = MdParameters::default();
    if let Some(v) = get_str("ensemble") { p.ensemble = v.to_lowercase(); }
    if let Some(v) = get_num("dt") { p.dt = v; }
    if let Some(v) = get_u64("steps") { p.steps = v as usize; }
    if let Some(v) = get_num("temperature") { p.temperature = v; }
    if let Some(v) = get_num("friction") { p.friction = v; }
    if let Some(v) = get_str("friction_units") { p.friction_units = v.to_lowercase(); }
    if let Some(v) = get_u64("seed") { p.seed = v; }
    if let Some(v) = get_str("init_velocities") { p.init_velocities = v.to_lowercase(); }
    if let Some(v) = get_str("velocities_file") { p.velocities_file = v; }
    if let Some(v) = get_str("restart_input") { p.restart_input = v; }
    if let Some(v) = get_u64("traj_interval") { p.traj_interval = v.max(1) as usize; }
    if let Some(v) = get_u64("equil_steps") { p.equil_steps = v as usize; }
    if let Some(v) = get_str("out_prefix") { p.out_prefix = v; }
    if let Some(v) = get_str("restart_output") { p.restart_output = v; }
    let mut outputs_explicit: Option<Vec<String>> = None;
    if merged.get("outputs").is_some() {
        let requested: Vec<String> = match merged.get("outputs").unwrap() {
            serde_json::Value::Array(items) => items
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_lowercase()))
                .collect(),
            serde_json::Value::String(s) =>
                s.split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|t| !t.is_empty())
                    .map(|t| t.to_lowercase())
                    .collect(),
            _ => return Err(anyhow::anyhow!(
                "md outputs must be a list or comma-separated string, e.g. [\"md_log\", \"traj\"]")),
        };
        for name in requested.iter() {
            if !OUTPUT_SWITCHES.contains(&name.as_str()) {
                return Err(anyhow::anyhow!(
                    "md outputs: unknown switch \"{}\" (valid: {})",
                    name, OUTPUT_SWITCHES.join(", ")));
            }
        }
        outputs_explicit = Some(requested);
    }

    let build_box = get_bool("qmmm_build_box").unwrap_or(false);
    let build_system = get_bool("qmmm_build_system").unwrap_or(false);
    let build_top = get_str("qmmm_top").map(|s| !s.is_empty()).unwrap_or(false);

    if let Some(mm_file) = get_str("qmmm_mm_file").filter(|s| !s.is_empty())
        .or_else(|| get_str("mm_file").filter(|s| !s.is_empty()))
        .or_else(|| build_box.then(String::new))
        .or_else(|| build_system.then(String::new))
        .or_else(|| build_top.then(String::new)) {

        let mut qmmm = QmmmParams::default();
        qmmm.mm_file = mm_file;
        qmmm.build_system = build_system;
        if let Some(v) = get_str("qmmm_water_model") {
            let lv = v.to_lowercase();
            if lv != "tip3p" {
                return Err(anyhow::anyhow!(
                    "qmmm_water_model: only \"tip3p\" is currently supported, got \"{}\"", v));
            }
            qmmm.water_model = lv;
        }
        if let Some(v) = get_num("qmmm_mm_cutoff") { qmmm.mm_cutoff = v; }
        if let Some(v) = get_num("mm_cutoff") { qmmm.mm_cutoff = v; }
        if let Some(v) = get_bool("qmmm_pbc") { qmmm.pbc = v; }
        if let Some(v) = get_str("qmmm_system_xml") { qmmm.system_xml = v; }

        if let Some(v) = get_str("mm_system_xml") { qmmm.system_xml = v; }
        if let Some(v) = get_num("qmmm_link_r_eq") { qmmm.link_r_eq = v; }
        if let Some(v) = get_num("qmmm_nb_cutoff") { qmmm.nb_cutoff = v; }
        if let Some(v) = get_u64("qmmm_gro_skip") { qmmm.gro_skip = v as usize; }
        if let Some(v) = get_bool("qmmm_build_box") { qmmm.build_box = v; }
        if let Some(v) = get_num("qmmm_box_margin") { qmmm.box_margin = v; }
        if let Some(v) = get_num("qmmm_box_spacing") { qmmm.box_spacing = v; }
        if let Some(v) = get_str("qmmm_mm_pdb") { qmmm.mm_pdb = v; }
        if let Some(v) = merged.get("qmmm_frontier") {
            qmmm.frontier = parse_usize_list(v);
        }
        if let Some(v) = get_str("qmmm_top") { qmmm.top = v; }
        if let Some(v) = get_str("qmmm_ff_dir") { qmmm.ff_dir = v; }
        if let Some(v) = merged.get("qmmm_qm_atoms") {
            qmmm.qm_atoms = parse_usize_list(v);
        }
        if let Some(v) = get_str("qmmm_frontier_rescale") {
            qmmm.frontier_rescale = v.to_lowercase();
        }
        if let Some(v) = merged.get("qmmm_ff") {
            let ffs: Vec<String> = match v {
                serde_json::Value::Array(items) => items
                    .iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect(),
                serde_json::Value::String(s) => s
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|t| !t.is_empty())
                    .map(|t| t.to_string())
                    .collect(),
                _ => return Err(anyhow::anyhow!(
                    "qmmm_ff must be a list or comma-separated string of OpenMM force-field files")),
            };
            qmmm.ff = ffs;
        }
        if merged.get("qmmm_links").is_some() {
            let mut links: Vec<(usize, usize)> = Vec::new();
            match merged.get("qmmm_links").unwrap() {
                serde_json::Value::Array(pairs) => {
                    for pair in pairs.iter() {
                        let idx = parse_usize_list(pair);
                        if idx.len() != 2 {
                            return Err(anyhow::anyhow!(
                                "qmmm_links entries must be [qm_bdry, mm_host] pairs, got {:?}",
                                pair));
                        }
                        links.push((idx[0], idx[1]));
                    }
                }
                _ => return Err(anyhow::anyhow!(
                    "qmmm_links must be a list of [qm_bdry, mm_host] pairs")),
            }
            qmmm.links = links;
        }
        if let Some(serde_json::Value::Object(lj)) = merged.get("qmmm_lj") {
            for (elem, params) in lj.iter() {
                let sigma_eps: Vec<f64> = match params {
                    serde_json::Value::Array(a) =>
                        a.iter().filter_map(|x| x.as_f64()).collect(),
                    serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
                    _ => vec![],
                };
                if sigma_eps.len() == 2 {
                    let elem_up = crate::geom_io::formated_element_name(&(elem.clone()));
                    qmmm.lj.insert(elem_up, [sigma_eps[0], sigma_eps[1]]);
                } else {
                    return Err(anyhow::anyhow!(
                        "qmmm_lj entry '{}' must be [sigma_A, epsilon_kJ/mol]", elem));
                }
            }
        }
        if qmmm.link_r_eq <= 0.0 {
            return Err(anyhow::anyhow!("qmmm_link_r_eq must be positive, got {}", qmmm.link_r_eq));
        }
        if !qmmm.top.is_empty() {
            if qmmm.mm_pdb.is_empty() && qmmm.mm_file.is_empty() {
                return Err(anyhow::anyhow!(
                    "qmmm_top requires qmmm_mm_pdb (PDB) or qmmm_mm_file (GRO, atoms in topology order)"));
            }
            if qmmm.ff_dir.is_empty() {
                return Err(anyhow::anyhow!(
                    "qmmm_top requires qmmm_ff_dir (GROMACS force-field directory)"));
            }
            if qmmm.qm_atoms.is_empty() && qmmm.gro_skip == 0 {
                return Err(anyhow::anyhow!(
                    "qmmm_top requires qmmm_qm_atoms or qmmm_gro_skip > 0"));
            }
        }
        if qmmm.build_system {
            if qmmm.ff.is_empty() {
                return Err(anyhow::anyhow!(
                    "qmmm_build_system = true requires qmmm_ff (OpenMM force-field files)"));
            }
            if qmmm.mm_pdb.is_empty() {
                return Err(anyhow::anyhow!(
                    "qmmm_build_system = true requires qmmm_mm_pdb (PDB with standard residues, QM atoms first)"));
            }
            if qmmm.gro_skip == 0 && qmmm.qm_atoms.is_empty() {
                return Err(anyhow::anyhow!(
                    "qmmm_build_system = true requires qmmm_qm_atoms or qmmm_gro_skip > 0 (QM region)"));
            }
        }
        p.qmmm = Some(qmmm);
    } else if get_str("qmmm_system_xml").map(|s| !s.is_empty()).unwrap_or(false)
        || get_str("mm_system_xml").map(|s| !s.is_empty()).unwrap_or(false)
        || get_str("qmmm_mm_pdb").map(|s| !s.is_empty()).unwrap_or(false)
    {
        return Err(anyhow::anyhow!(
            "QM/MM: qmmm_system_xml/mm_system_xml/qmmm_mm_pdb requires qmmm_mm_file (MM positions); \
             set qmmm_mm_file, or qmmm_build_system/qmmm_top/qmmm_build_box"));
    }

    p.outputs = outputs_explicit.unwrap_or_else(|| default_outputs_for(p.qmmm.is_some()));

    if merged.get("umbrella_atoms").is_some() {
        let mut umb = UmbrellaParams::default();
        umb.atoms = parse_usize_list(merged.get("umbrella_atoms").unwrap());
        if let Some(v) = get_num("umbrella_center") { umb.center = v; }
        if let Some(v) = get_num("umbrella_kappa") { umb.kappa = v; }
        if let Some(v) = get_str("umbrella_potential") { umb.potential = v.to_lowercase(); }
        if let Some(v) = get_str("umbrella_cv") { umb.cv = v.to_lowercase(); }
        if let Some(v) = get_num("umbrella_sum_kappa") { umb.sum_kappa = v; }
        if let Some(v) = get_num("umbrella_sum_center") { umb.sum_center = v; }
        p.umbrella = Some(umb);
    }

    if !p.ensemble.eq("nvt")
        && !p.ensemble.eq("nve")
        && !p.ensemble.eq("opt")
        && !p.ensemble.eq("sp")
        && !p.ensemble.eq("singlepoint")
    {
        return Err(anyhow::anyhow!(
            "md ensemble must be \"nvt\", \"nve\", \"opt\" or \"sp\", got \"{}\"", p.ensemble));
    }
    if p.opt_fmax <= 0.0 {
        return Err(anyhow::anyhow!("md opt_fmax must be positive, got {}", p.opt_fmax));
    }
    if p.dt <= 0.0 {
        return Err(anyhow::anyhow!("md dt must be positive, got {}", p.dt));
    }
    if p.steps == 0 {
        return Err(anyhow::anyhow!("md steps must be positive, got {}", p.steps));
    }
    if p.ensemble.eq("nvt") && p.friction < 0.0 {
        return Err(anyhow::anyhow!("md friction must be non-negative, got {}", p.friction));
    }
    if !p.friction_units.eq("ase") && !p.friction_units.eq("fs_inv") {
        return Err(anyhow::anyhow!(
            "md friction_units must be \"ase\" or \"fs_inv\", got \"{}\"", p.friction_units));
    }
    if !p.init_velocities.eq("maxwell") && !p.init_velocities.eq("file") && !p.init_velocities.eq("zero") {
        return Err(anyhow::anyhow!(
            "md init_velocities must be \"maxwell\", \"file\" or \"zero\", got \"{}\"",
            p.init_velocities));
    }
    if p.init_velocities.eq("file") && p.velocities_file.is_empty() {
        return Err(anyhow::anyhow!("md init_velocities = \"file\" requires velocities_file"));
    }
    if let Some(umb) = &p.umbrella {
        if !matches!(umb.cv.as_str(), "dihedral" | "distance" | "angle" | "distance_diff" | "distance difference") {
            return Err(anyhow::anyhow!(
                "umbrella_cv must be \"dihedral\", \"distance\", \"angle\" or \"distance_diff\", got \"{}\"", umb.cv));
        }
        let need = super::umbrella::CvType::n_atoms(umb.cv_type());
        if umb.atoms.len() != need {
            return Err(anyhow::anyhow!(
                "umbrella_cv = \"{}\" needs exactly {} atom indices, got {:?}",
                umb.cv, need, umb.atoms));
        }
        if !umb.potential.eq("cosine") && !umb.potential.eq("harmonic") {
            return Err(anyhow::anyhow!(
                "umbrella_potential must be \"cosine\" or \"harmonic\", got \"{}\"",
                umb.potential));
        }
        if umb.kappa < 0.0 {
            return Err(anyhow::anyhow!("umbrella_kappa must be non-negative, got {}", umb.kappa));
        }
        if umb.sum_kappa < 0.0 {
            return Err(anyhow::anyhow!(
                "umbrella_sum_kappa must be non-negative, got {}",
                umb.sum_kappa
            ));
        }
        if umb.sum_kappa > 0.0 && umb.cv_type() != super::umbrella::CvType::DistanceDiff {
            return Err(anyhow::anyhow!(
                "umbrella_sum_kappa is only supported with umbrella_cv = \"distance_diff\""
            ));
        }
    }

    Ok(Some(p))
}
