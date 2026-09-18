
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const DEG: f64 = std::f64::consts::PI / 180.0;

type DihedralKey = (String, String, String, String);

#[derive(Default, Clone)]
struct Molecule {
    name: String,
    types: Vec<String>,
    charges: Vec<f64>,
    masses: Vec<f64>,
    resnames: Vec<String>,
    atomnames: Vec<String>,
    resids: Vec<i64>,
    bonds: Vec<(usize, usize, f64, f64)>,
    angles: Vec<(usize, usize, usize, f64, f64, f64, f64)>,
    periodic: Vec<(usize, usize, usize, usize, f64, f64, u32)>,
    harmonic: Vec<(usize, usize, usize, usize, f64, f64)>,
    rb: Vec<(usize, usize, usize, usize, [f64; 6])>,
    pairs: Vec<(usize, usize)>,
    cmaps: Vec<(usize, usize, usize, usize, usize, CmapKey)>,
    exclusions: Vec<(usize, usize)>,
    ub: Vec<(usize, usize, f64, f64)>,
}

type CmapKey = (String, String, String, String, String);

#[derive(Default)]
struct ForceField {
    atomtypes: BTreeMap<String, (f64, f64)>,
    pairtypes: BTreeMap<(String, String), (f64, f64)>,
    bondtypes: BTreeMap<(String, String), (f64, f64)>,
    angletypes: BTreeMap<(String, String, String), [f64; 4]>,
    dihedraltypes: BTreeMap<DihedralKey, Vec<Vec<String>>>,
    cmaptypes: BTreeMap<CmapKey, Vec<f64>>,
    nbfix: BTreeMap<(String, String), (f64, f64)>,
    fudge_lj: f64,
    fudge_qq: f64,
    gen_pairs: bool,
}

fn load_defaults(sections: &[(String, Vec<Vec<String>>)]) -> Option<(f64, f64, bool)> {
    sec(sections, "defaults").and_then(|rows| rows.first()).map(|f| {
        let gen = f.get(2).map(|s| s.eq_ignore_ascii_case("yes")).unwrap_or(false);
        let flj = f.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let fqq = f.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        (flj, fqq, gen)
    })
}

fn gen_pairs_for_mol(m: &Molecule) -> Vec<(usize, usize)> {
    let n = m.types.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut bonded: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    for (a, b, _, _) in &m.bonds {
        adj[*a].push(*b);
        adj[*b].push(*a);
        bonded.insert(((*a).min(*b), (*a).max(*b)));
    }
    let mut existing: std::collections::BTreeSet<(usize, usize)> =
        std::collections::BTreeSet::new();
    for (a, b) in &m.pairs {
        existing.insert(((*a).min(*b), (*a).max(*b)));
    }
    let mut out: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    for (b, c, _, _) in &m.bonds {
        for &a in &adj[*b] {
            if a == *c {
                continue;
            }
            for &d in &adj[*c] {
                if d == *b || d == a {
                    continue;
                }
                let key = (a.min(d), a.max(d));
                if bonded.contains(&key) {
                    continue;
                }
                if bonded.contains(&(a.min(*c), a.max(*c))) {
                    continue;
                }
                if bonded.contains(&((*b).min(d), (*b).max(d))) {
                    continue;
                }
                if existing.contains(&key) {
                    continue;
                }
                out.insert(key);
            }
        }
    }
    out.into_iter().collect()
}

fn read_text(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(_) => String::new(),
    }
}

fn skip_comment(line: &str) -> String {
    line.split(';').next().unwrap_or("").trim().to_string()
}

fn read_sections(path: &Path, resolve: bool) -> Vec<(String, Vec<Vec<String>>)> {
    let text = read_text(path);
    let mut out: Vec<(String, Vec<Vec<String>>)> = Vec::new();
    let mut cur: Option<(String, Vec<Vec<String>>)> = None;
    for raw in text.lines() {
        let line = skip_comment(raw);
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if resolve && line.starts_with("#include") {
                let inc = if let Some(i) = line.find('"') {
                    line[i + 1..].split('"').next().unwrap_or("").to_string()
                } else {
                    line.split_whitespace().nth(1).unwrap_or("").to_string()
                };
                if let Some(c) = cur.take() {
                    out.push(c);
                }
                let p = path.parent().unwrap_or(Path::new(".")).join(inc);
                out.extend(read_sections(&p, true));
            }
            continue;
        }
        if line.starts_with('[') {
            if let Some(c) = cur.take() {
                out.push(c);
            }
            let name = line
                .trim_matches(|c: char| c == '[' || c == ']' || c.is_whitespace())
                .to_lowercase();
            cur = Some((name, Vec::new()));
        } else if let Some((_, rows)) = cur.as_mut() {
            rows.push(line.split_whitespace().map(|s| s.to_string()).collect());
        }
    }
    if let Some(c) = cur.take() {
        out.push(c);
    }
    out
}

fn sec<'a>(sections: &'a [(String, Vec<Vec<String>>)], name: &str) -> Option<&'a Vec<Vec<String>>> {
    sections.iter().find(|(n, _)| n == name).map(|(_, r)| r)
}

fn secs<'a>(sections: &'a [(String, Vec<Vec<String>>)], name: &str) -> Vec<&'a Vec<Vec<String>>> {
    sections.iter().filter(|(n, _)| n == name).map(|(_, r)| r).collect()
}

impl ForceField {
    fn load(ffdir: &Path) -> Self {
        let mut ff = ForceField::default();
        ff.fudge_lj = 1.0;
        ff.fudge_qq = 1.0;
        if let Some(rows) = sec(&read_sections(&ffdir.join("ffnonbonded.itp"), false), "atomtypes") {
            for f in rows {
                if f.len() >= 2 {
                    if let (Ok(s), Ok(e)) = (f[f.len() - 2].parse(), f[f.len() - 1].parse()) {
                        ff.atomtypes.insert(f[0].clone(), (s, e));
                    }
                }
            }
        }
        if let Some(rows) = sec(&read_sections(&ffdir.join("ffnonbonded.itp"), false), "pairtypes") {
            for f in rows {
                if f.len() >= 5 {
                    if let (Ok(s), Ok(e)) = (f[3].parse(), f[4].parse()) {
                        ff.pairtypes.insert((f[0].clone(), f[1].clone()), (s, e));
                        ff.pairtypes.insert((f[1].clone(), f[0].clone()), (s, e));
                    }
                }
            }
        }
        let bonded = read_sections(&ffdir.join("ffbonded.itp"), true);
        if let Some(rows) = sec(&bonded, "bondtypes") {
            for f in rows {
                if f.len() >= 5 {
                    if let (Ok(r), Ok(k)) = (f[3].parse(), f[4].parse()) {
                        let key = if f[0] <= f[1] {
                            (f[0].clone(), f[1].clone())
                        } else {
                            (f[1].clone(), f[0].clone())
                        };
                        ff.bondtypes.insert(key, (r, k));
                    }
                }
            }
        }
        if let Some(rows) = sec(&bonded, "angletypes") {
            for f in rows {
                if f.len() >= 6 {
                    let th: f64 = f[4].parse().unwrap_or(0.0);
                    let k: f64 = f[5].parse().unwrap_or(0.0);
                    let ub0: f64 = f.get(6).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                    let kub: f64 = f.get(7).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                    ff.angletypes.insert(
                        (f[0].clone(), f[1].clone(), f[2].clone()),
                        [th, k, ub0, kub],
                    );
                }
            }
        }
        for rows in secs(&bonded, "dihedraltypes") {
            for f in rows {
                if f.len() >= 5 && f[4].parse::<u32>().is_ok() {
                    ff.dihedraltypes
                        .entry((f[0].clone(), f[1].clone(), f[2].clone(), f[3].clone()))
                        .or_default()
                        .push(f.clone());
                }
            }
        }
        ff.cmaptypes = load_cmaptypes(ffdir);
        ff.nbfix = load_nbfix(ffdir);
        if let Some((flj, fqq, gen)) =
            load_defaults(&read_sections(&ffdir.join("forcefield.itp"), false))
        {
            ff.fudge_lj = flj;
            ff.fudge_qq = fqq;
            ff.gen_pairs = gen;
        }
        ff
    }
}

fn load_cmaptypes(ffdir: &Path) -> BTreeMap<CmapKey, Vec<f64>> {
    let p = ffdir.join("cmap.itp");
    let mut out = BTreeMap::new();
    if !p.exists() {
        return out;
    }
    let mut toks: Vec<String> = Vec::new();
    for raw in read_text(&p).lines() {
        let mut ln = raw.split(';').next().unwrap_or("").trim_end_matches('\\').trim().to_string();
        if ln.is_empty() || ln.starts_with('[') || ln.starts_with('#') {
            continue;
        }
        ln = ln.split_whitespace().collect::<Vec<_>>().join(" ");
        toks.extend(ln.split_whitespace().map(|s| s.to_string()));
    }
    let mut i = 0usize;
    while i + 8 <= toks.len() {
        if toks[i + 5] == "1" && toks[i + 6] == toks[i + 7] {
            if let Ok(n) = toks[i + 6].parse::<usize>() {
                let start = i + 8;
                if start + n * n <= toks.len() {
                    let key: CmapKey = (
                        toks[i].clone(),
                        toks[i + 1].clone(),
                        toks[i + 2].clone(),
                        toks[i + 3].clone(),
                        toks[i + 4].clone(),
                    );
                    let grid: Vec<f64> =
                        toks[start..start + n * n].iter().filter_map(|x| x.parse().ok()).collect();
                    out.insert(key, grid);
                    i += 8 + n * n;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

fn load_nbfix(ffdir: &Path) -> BTreeMap<(String, String), (f64, f64)> {
    let mut out = BTreeMap::new();
    for name in ["nbfix.itp", "ffnonbonded.itp"] {
        let p = ffdir.join(name);
        if !p.exists() {
            continue;
        }
        if let Some(rows) = sec(&read_sections(&p, false), "nonbond_params") {
            for f in rows {
                if f.len() >= 5 {
                    if let (Ok(s), Ok(e)) = (f[3].parse(), f[4].parse()) {
                        let key = if f[0] <= f[1] {
                            (f[0].clone(), f[1].clone())
                        } else {
                            (f[1].clone(), f[0].clone())
                        };
                        out.insert(key, (s, e));
                    }
                }
            }
        }
    }
    out
}

fn included_files(top: &Path) -> Vec<PathBuf> {
    let mut seen: Vec<PathBuf> = Vec::new();
    fn walk(p: PathBuf, seen: &mut Vec<PathBuf>) {
        let canon = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
        if seen.contains(&canon) || !p.exists() {
            return;
        }
        seen.push(canon);
        for raw in read_text(&p).lines() {
            let line = skip_comment(raw);
            if line.starts_with("#include") {
                let inc = if let Some(i) = line.find('"') {
                    line[i + 1..].split('"').next().unwrap_or("").to_string()
                } else {
                    line.split_whitespace().nth(1).unwrap_or("").to_string()
                };
                walk(p.parent().unwrap_or(Path::new(".")).join(inc), seen);
            }
        }
    }
    walk(top.to_path_buf(), &mut seen);
    seen
}

fn idx(x: &str) -> usize {
    x.parse::<usize>().unwrap_or(1).saturating_sub(1)
}

fn match_dihedral(ff: &ForceField, types: &DihedralKey) -> Option<Vec<Vec<String>>> {
    let rev: DihedralKey = (types.3.clone(), types.2.clone(), types.1.clone(), types.0.clone());
    let mut wildcard: Option<Vec<Vec<String>>> = None;
    for (key, rows) in ff.dihedraltypes.iter() {
        let forward = (0..4).all(|k| {
            let kk = [&key.0, &key.1, &key.2, &key.3][k];
            kk == "X" || kk == [&types.0, &types.1, &types.2, &types.3][k]
        });
        let reverse = (0..4).all(|k| {
            let kk = [&key.0, &key.1, &key.2, &key.3][k];
            kk == "X" || kk == [&rev.0, &rev.1, &rev.2, &rev.3][k]
        });
        if forward || reverse {
            if !key.0.eq("X") && !key.1.eq("X") && !key.2.eq("X") && !key.3.eq("X") {
                return Some(rows.clone());
            }
            if wildcard.is_none() {
                wildcard = Some(rows.clone());
            }
        }
    }
    wildcard
}

fn parse_molecules(path: &Path, ff: &ForceField) -> Vec<Molecule> {
    let sections = read_sections(path, false);
    let mut mols: Vec<Molecule> = Vec::new();
    let mut cur: Option<Molecule> = None;
    for (name, rows) in sections.iter() {
        if name == "moleculetype" {
            if let Some(m) = cur.take() {
                mols.push(m);
            }
            let mut m = Molecule::default();
            m.name = rows.first().and_then(|r| r.first()).cloned().unwrap_or_default();
            cur = Some(m);
        } else if let Some(m) = cur.as_mut() {
            if name == "atoms" {
                for f in rows {
                    if f.len() < 7 {
                        continue;
                    }
                    m.types.push(f[1].clone());
                    m.resids.push(f[2].parse().unwrap_or(0));
                    m.resnames.push(f[3].clone());
                    m.atomnames.push(f[4].clone());
                    m.charges.push(f[6].parse().unwrap_or(0.0));
                    m.masses.push(f.get(7).and_then(|x| x.parse().ok()).unwrap_or(0.0));
                }
            }
        }
    }
    if let Some(m) = cur.take() {
        mols.push(m);
    }

    for mol in mols.iter_mut() {
        let sections = read_sections(path, false);
        let mut blocks: HashMap<String, Vec<Vec<Vec<String>>>> = HashMap::new();
        let mut active = false;
        for (name, rows) in sections.iter() {
            if name == "moleculetype" {
                active = rows.first().and_then(|r| r.first()).map(|n| n == &mol.name).unwrap_or(false);
                continue;
            }
            if active
                && ["atoms", "bonds", "angles", "dihedrals", "pairs", "cmap", "exclusions"]
                    .contains(&name.as_str())
            {
                blocks.entry(name.clone()).or_default().push(rows.clone());
            }
        }
        let t = &mol.types;
        let push = |v: &mut Vec<Vec<String>>, rows: &Vec<Vec<String>>| v.extend(rows.clone());

        let mut bro: Vec<Vec<String>> = Vec::new();
        if let Some(b) = blocks.get("bonds") {
            for r in b {
                push(&mut bro, r);
            }
        }
        for f in &bro {
            if f.len() < 2 {
                continue;
            }
            let (i, j) = (idx(&f[0]), idx(&f[1]));
            if f.len() >= 5 {
                mol.bonds.push((i, j, f[3].parse().unwrap_or(0.0), f[4].parse().unwrap_or(0.0)));
            } else {
                let key = if t[i] <= t[j] { (t[i].clone(), t[j].clone()) } else { (t[j].clone(), t[i].clone()) };
                if let Some((r0, k)) = ff.bondtypes.get(&key) {
                    mol.bonds.push((i, j, *r0, *k));
                } else {
                    return_mol_err(path, "bondtype", &format!("{:?}", key));
                }
            }
        }
        let mut ub: Vec<(usize, usize, f64, f64)> = Vec::new();
        let mut aro: Vec<Vec<String>> = Vec::new();
        if let Some(b) = blocks.get("angles") {
            for r in b {
                push(&mut aro, r);
            }
        }
        for f in &aro {
            if f.len() < 3 {
                continue;
            }
            let (i, j, k) = (idx(&f[0]), idx(&f[1]), idx(&f[2]));
            let funct: u32 = f.get(3).and_then(|x| x.parse().ok()).unwrap_or(1);
            if funct != 1 && funct != 5 {
                return_mol_err(path, "angle funct", &funct.to_string());
            }
            if f.len() >= 6 {
                let th0: f64 = f[4].parse().unwrap_or(0.0);
                let kth: f64 = f[5].parse().unwrap_or(0.0);
                let ub0: f64 = f.get(6).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                let kub: f64 = f.get(7).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                mol.angles.push((i, j, k, th0, kth, ub0, kub));
                if kub > 0.0 { ub.push((i, k, ub0, kub)); }
            } else {
                let key = (t[i].clone(), t[j].clone(), t[k].clone());
                let key = if ff.angletypes.contains_key(&key) {
                    key
                } else {
                    (key.2, key.1, key.0)
                };
                if let Some(v) = ff.angletypes.get(&key) {
                    mol.angles.push((i, j, k, v[0], v[1], v[2], v[3]));
                    if v[3] > 0.0 { ub.push((i, k, v[2], v[3])); }
                } else {
                    return_mol_err(path, "angletype", &format!("{:?}", key));
                }
            }
        }
        let mut dro: Vec<Vec<String>> = Vec::new();
        if let Some(b) = blocks.get("dihedrals") {
            for r in b {
                push(&mut dro, r);
            }
        }
        for f in &dro {
            if f.len() < 5 {
                continue;
            }
            let (i, j, k, l) = (idx(&f[0]), idx(&f[1]), idx(&f[2]), idx(&f[3]));
            let funct: u32 = f[4].parse().unwrap_or(0);
            let explicit = ((matches!(funct, 1 | 4 | 5 | 9) && f.len() > 7)
                || (funct == 3 && f.len() > 10)
                || (funct == 2 && f.len() > 6)
                || (funct == 8 && f.len() > 10));
            let params_list: Vec<Vec<String>> = if explicit {
                vec![f.clone()]
            } else {
                let key = (t[i].clone(), t[j].clone(), t[k].clone(), t[l].clone());
                match match_dihedral(ff, &key) {
                    Some(v) => v,
                    None => return_mol_err(path, "dihedraltype", &format!("{:?}", key)),
                }
            };
            for p in params_list.iter() {
                match funct {
                    1 | 4 | 9 => {
                        let kk: f64 = p.get(6).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                        if kk != 0.0 {
                            mol.periodic.push((
                                i,
                                j,
                                k,
                                l,
                                p.get(5).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) * DEG,
                                kk,
                                p.get(7).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) as u32,
                            ));
                        }
                    }
                    2 => {
                        let kk: f64 = p.get(6).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                        if kk != 0.0 {
                            mol.harmonic.push((
                                i,
                                j,
                                k,
                                l,
                                p.get(5).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) * DEG,
                                kk,
                            ));
                        }
                    }
                    3 | 8 => {
                        let c = parse6(p);
                        if c.iter().any(|x| *x != 0.0) {
                            mol.rb.push((i, j, k, l, c));
                        }
                    }
                    5 => {
                        let c0 = parse6(p);
                        let c = [
                            c0[1] + 0.5 * (c0[0] + c0[2]),
                            0.5 * (-c0[0] + 3.0 * c0[2]),
                            -c0[1] + 4.0 * c0[3],
                            -2.0 * c0[2],
                            -4.0 * c0[3],
                            0.0,
                        ];
                        if c.iter().any(|x| *x != 0.0) {
                            mol.rb.push((i, j, k, l, c));
                        }
                    }
                    _ => return_mol_err(path, "dihedral funct", &funct.to_string()),
                }
            }
        }
        if let Some(b) = blocks.get("pairs") {
            for r in b {
                for f in r {
                    if f.len() >= 2 {
                        mol.pairs.push((idx(&f[0]), idx(&f[1])));
                    }
                }
            }
        }
        if let Some(b) = blocks.get("cmap") {
            for r in b {
                for f in r {
                    if f.len() < 5 {
                        continue;
                    }
                    let ids = [idx(&f[0]), idx(&f[1]), idx(&f[2]), idx(&f[3]), idx(&f[4])];
                    let key: CmapKey = (
                        t[ids[0]].clone(),
                        t[ids[1]].clone(),
                        t[ids[2]].clone(),
                        t[ids[3]].clone(),
                        t[ids[4]].clone(),
                    );
                    if !ff.cmaptypes.contains_key(&key) {
                        return_mol_err(path, "cmaptype", &format!("{:?}", key));
                    }
                    mol.cmaps.push((ids[0], ids[1], ids[2], ids[3], ids[4], key));
                }
            }
        }
        if let Some(b) = blocks.get("exclusions") {
            for r in b {
                for f in r {
                    if f.is_empty() {
                        continue;
                    }
                    let a = idx(&f[0]);
                    for x in &f[1..] {
                        mol.exclusions.push((a, idx(x)));
                    }
                }
            }
        }
    }
    mols
}

fn parse6(p: &[String]) -> [f64; 6] {
    let mut c = [0.0; 6];
    for k in 0..6 {
        c[k] = p.get(5 + k).and_then(|x| x.parse().ok()).unwrap_or(0.0);
    }
    c
}

fn return_mol_err(_p: &Path, _what: &str, _key: &str) -> ! {
    panic!("force-field topology: missing {} {}", _what, _key)
}

struct Structure {
    positions: Vec<[f64; 3]>,
    resnames: Vec<String>,
    resids: Vec<i64>,
    atomnames: Vec<String>,
    elements: Vec<String>,
    box_nm: [f64; 3],
}

fn parse_pdb(path: &Path) -> Structure {
    let mut s = Structure {
        positions: Vec::new(),
        resnames: Vec::new(),
        resids: Vec::new(),
        atomnames: Vec::new(),
        elements: Vec::new(),
        box_nm: [0.0; 3],
    };
    for raw in read_text(path).lines() {
        if raw.starts_with("CRYST1") {
            let f: Vec<&str> = raw.split_whitespace().collect();
            if f.len() >= 4 {
                s.box_nm = [
                    f[1].parse::<f64>().unwrap_or(0.0) / 10.0,
                    f[2].parse::<f64>().unwrap_or(0.0) / 10.0,
                    f[3].parse::<f64>().unwrap_or(0.0) / 10.0,
                ];
            }
        }
        if !(raw.starts_with("ATOM") || raw.starts_with("HETATM")) {
            continue;
        }
        s.atomnames.push(raw.get(12..16).unwrap_or("").trim().to_string());
        s.resnames.push(raw.get(17..20).unwrap_or("").trim().to_string());
        s.resids.push(raw.get(22..26).unwrap_or("").trim().parse().unwrap_or_else(|_| {
            i64::from_str_radix(raw.get(22..26).unwrap_or("").trim(), 16).unwrap_or(0)
        }));
        let el = raw.get(76..78).unwrap_or("").trim().to_string();
        let el = if !el.is_empty() {
            el
        } else {
            let nm = s.atomnames.last().cloned().unwrap_or_default();
            let up = nm.to_uppercase();
            match up.as_str() {
                "CL" => "Cl".to_string(),
                "NA" => "Na".to_string(),
                "MG" => "Mg".to_string(),
                "CA" => "Ca".to_string(),
                _ => nm.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_else(|| "X".to_string()),
            }
        };
        s.elements.push(el);
        let g = |a: usize, b: usize| raw.get(a..b).unwrap_or("").trim().parse::<f64>().unwrap_or(0.0);
        s.positions.push([g(30, 38), g(38, 46), g(46, 54)]);
    }
    s
}

fn parse_gro(path: &Path) -> Structure {
    let text = read_text(path);
    let lines: Vec<&str> = text.lines().collect();
    let n: usize = lines.get(1).and_then(|l| l.trim().parse().ok()).unwrap_or(0);
    let mut s = Structure {
        positions: Vec::new(),
        resnames: Vec::new(),
        resids: Vec::new(),
        atomnames: Vec::new(),
        elements: Vec::new(),
        box_nm: [0.0; 3],
    };
    for k in 0..n {
        let ln = lines[2 + k];
        s.resnames.push(ln.get(5..10).unwrap_or("").trim().to_string());
        let nm = ln.get(10..15).unwrap_or("").trim().to_string();
        s.atomnames.push(nm.clone());
        s.resids.push(ln.get(0..5).unwrap_or("").trim().parse().unwrap_or(0));
        let up = nm.to_uppercase();
        let el = if up.len() >= 2 && matches!(&up[..2], "CL" | "NA" | "MG" | "CA" | "ZN" | "FE")
            && !nm.chars().nth(1).map(|c| c.is_lowercase()).unwrap_or(false)
        {
            let mut c = up[..2].to_lowercase();
            c[0..1].make_ascii_uppercase();
            c
        } else {
            nm.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_else(|| "X".to_string())
        };
        s.elements.push(el);
        let g = |a: usize, b: usize| ln.get(a..b).unwrap_or("").trim().parse::<f64>().unwrap_or(0.0);
        s.positions.push([g(20, 28) * 10.0, g(28, 36) * 10.0, g(36, 44) * 10.0]);
    }
    if let Some(bl) = lines.get(2 + n) {
        let f: Vec<&str> = bl.split_whitespace().collect();
        if f.len() >= 3 {
            s.box_nm = [
                f[0].parse::<f64>().unwrap_or(0.0),
                f[1].parse::<f64>().unwrap_or(0.0),
                f[2].parse::<f64>().unwrap_or(0.0),
            ];
        }
    }
    s
}

struct ExceptionTable {
    keys: Vec<(usize, usize)>,
    vals: HashMap<(usize, usize), (f64, f64, f64)>,
    index: HashMap<(usize, usize), usize>,
}

impl ExceptionTable {
    fn new() -> Self {
        Self { keys: Vec::new(), vals: HashMap::new(), index: HashMap::new() }
    }
    fn set(&mut self, i: usize, j: usize, q: f64, s: f64, e: f64) {
        let key = (i.min(j), i.max(j));
        if let Some(&k) = self.index.get(&key) {
            self.vals.insert(key, (q, s, e));
            let _ = k;
        } else {
            self.index.insert(key, self.keys.len());
            self.keys.push(key);
            self.vals.insert(key, (q, s, e));
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_system_from_top(
    pdb_path: &str,
    top_path: &str,
    ff_dir: &str,
    qm_atoms: &[usize],
    links: &[(usize, usize)],
    frontier: &[usize],
    rescale_mode: &str,
    cutoff_nm: f64,
    out_xml: &str,
    out_gro: &str,
    gro_path: &str,
) -> anyhow::Result<usize> {
    let keep_order = !gro_path.is_empty();
    let struct_ = if keep_order { parse_gro(Path::new(gro_path)) } else { parse_pdb(Path::new(pdb_path)) };
    let n = struct_.positions.len();
    let mut ff = ForceField::load(Path::new(ff_dir));
    if let Some((flj, fqq, gen)) = load_defaults(&read_sections(Path::new(top_path), true)) {
        ff.fudge_lj = flj;
        ff.fudge_qq = fqq;
        ff.gen_pairs = gen;
    }

    let mut mols: BTreeMap<String, Molecule> = BTreeMap::new();
    for inc in included_files(Path::new(top_path)) {
        for m in parse_molecules(&inc, &ff) {
            mols.insert(m.name.clone(), m);
        }
    }
    if !mols.contains_key("SOL") {
        mols.insert("SOL".to_string(), make_tip3p());
    }

    let order_spec = sec(&read_sections(Path::new(top_path), false), "molecules")
        .cloned()
        .unwrap_or_default();
    let mut types: Vec<String> = Vec::new();
    let mut charges: Vec<f64> = Vec::new();
    let mut masses: Vec<f64> = Vec::new();
    let mut bonds: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut angles: Vec<(usize, usize, usize, f64, f64, f64, f64)> = Vec::new();
    let mut periodic: Vec<(usize, usize, usize, usize, f64, f64, u32)> = Vec::new();
    let mut harmonic: Vec<(usize, usize, usize, usize, f64, f64)> = Vec::new();
    let mut rb: Vec<(usize, usize, usize, usize, [f64; 6])> = Vec::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut gen_pairs: Vec<(usize, usize)> = Vec::new();
    let mut cmaps: Vec<(usize, usize, usize, usize, usize, CmapKey)> = Vec::new();
    let mut exclusions: Vec<(usize, usize)> = Vec::new();
    let mut ub: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut off = 0usize;
    for f in order_spec.iter() {
        if f.len() < 2 {
            continue;
        }
        let count: usize = f[1].parse().unwrap_or(0);
        let m = mols.get(&f[0]).ok_or_else(|| anyhow::anyhow!("topology molecule {} not found", f[0]))?;
        for _ in 0..count {
            types.extend(m.types.clone());
            charges.extend(m.charges.clone());
            masses.extend(m.masses.clone());
            bonds.extend(m.bonds.iter().map(|(a, b, r, k)| (a + off, b + off, *r, *k)));
            angles.extend(m.angles.iter().map(|(a, b, c, t, kt, u, ku)| (a + off, b + off, c + off, *t, *kt, *u, *ku)));
            periodic.extend(m.periodic.iter().map(|(a, b, c, d, p, k, mlt)| (a + off, b + off, c + off, d + off, *p, *k, *mlt)));
            harmonic.extend(m.harmonic.iter().map(|(a, b, c, d, p, k)| (a + off, b + off, c + off, d + off, *p, *k)));
            rb.extend(m.rb.iter().map(|(a, b, c, d, cs)| (a + off, b + off, c + off, d + off, *cs)));
            pairs.extend(m.pairs.iter().map(|(a, b)| (a + off, b + off)));
            if m.pairs.is_empty() && ff.gen_pairs {
                gen_pairs.extend(gen_pairs_for_mol(m).iter().map(|(a, b)| (a + off, b + off)));
            }
            cmaps.extend(m.cmaps.iter().map(|(a, b, c, d, e, key)| (a + off, b + off, c + off, d + off, e + off, key.clone())));
            exclusions.extend(m.exclusions.iter().map(|(a, b)| (a + off, b + off)));
            ub.extend(m.ub.iter().map(|(a, b, r, k)| (a + off, b + off, *r, *k)));
            off += m.types.len();
        }
    }
    if off != n {
        anyhow::bail!("assembled {} atoms vs structure {}", off, n);
    }

    let qm_set: std::collections::BTreeSet<usize> = qm_atoms.iter().cloned().collect();
    let mut fr: std::collections::BTreeSet<usize> = frontier.iter().cloned().collect();
    for (_, mh) in links {
        fr.insert(*mh);
    }
    for i in &qm_set {
        fr.remove(i);
    }

    let mut finalq = charges.clone();
    let mut by_res: BTreeMap<(String, i64), Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        by_res.entry((struct_.resnames[i].clone(), struct_.resids[i])).or_default().push(i);
    }
    if !fr.is_empty() {
        let mut removed: BTreeMap<(String, i64), f64> = BTreeMap::new();
        for i in &fr {
            removed.entry((struct_.resnames[*i].clone(), struct_.resids[*i])).and_modify(|x| *x += finalq[*i]).or_insert(finalq[*i]);
            finalq[*i] = 0.0;
        }
        if rescale_mode == "residue" {
            for (res, rest_all) in by_res.iter() {
                let rest: Vec<usize> =
                    rest_all.iter().cloned().filter(|i| !fr.contains(i) && !qm_set.contains(i)).collect();
                let r = removed.get(res).cloned().unwrap_or(0.0);
                if !rest.is_empty() && r.abs() > 1.0e-12 {
                    let share = r / rest.len() as f64;
                    for i in rest {
                        finalq[i] += share;
                    }
                }
            }
        }
    }
    for i in &qm_set {
        finalq[*i] = 0.0;
    }

    let mut tables = ExceptionTable::new();
    let mut excl12: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    let mut excl13: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    for (a, b, _, _) in &bonds {
        excl12.insert(((*a).min(*b), (*a).max(*b)));
    }
    for (a, _, c, _, _, _, _) in &angles {
        excl13.insert(((*a).min(*c), (*a).max(*c)));
    }
    for (a, b) in &exclusions {
        excl12.insert(((*a).min(*b), (*a).max(*b)));
    }
    let pair_lj = |i: usize, j: usize| -> (f64, f64) {
        ff.pairtypes
            .get(&(types[i].clone(), types[j].clone()))
            .cloned()
            .unwrap_or_else(|| {
                let (si, ei) = ff.atomtypes[&types[i]];
                let (sj, ej) = ff.atomtypes[&types[j]];
                (0.5 * (si + sj), (ei * ej).sqrt())
            })
    };
    for k in excl12.iter().chain(excl13.iter()) {
        let (i, j) = (k.0, k.1);
        let qi = qm_set.contains(&i);
        let qj = qm_set.contains(&j);
        if qi && qj {
            tables.set(i, j, 0.0, 1.0, 0.0);
        } else if qi || qj {
            let (sig, eps) = pair_lj(i, j);
            tables.set(i, j, 0.0, sig, eps);
        } else {
            tables.set(i, j, 0.0, 1.0, 0.0);
        }
    }
    for (i, j) in &pairs {
        let (sig, eps) = pair_lj(*i, *j);
        let q = if qm_set.contains(i) || qm_set.contains(j) { 0.0 } else { charges[*i] * charges[*j] };
        tables.set(*i, *j, q, sig, eps);
    }
    for (i, j) in &gen_pairs {
        let has_pt = ff.pairtypes.contains_key(&(types[*i].clone(), types[*j].clone()));
        let (sig, eps) = if has_pt {
            ff.pairtypes
                .get(&(types[*i].clone(), types[*j].clone()))
                .cloned()
                .unwrap_or((0.0, 0.0))
        } else {
            let (si, ei) = ff.atomtypes[&types[*i]];
            let (sj, ej) = ff.atomtypes[&types[*j]];
            (0.5 * (si + sj), (ei * ej).sqrt() * ff.fudge_lj)
        };
        let q = if qm_set.contains(i) || qm_set.contains(j) {
            0.0
        } else {
            charges[*i] * charges[*j] * ff.fudge_qq
        };
        tables.set(*i, *j, q, sig, eps);
    }
    let mut by_type: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, t) in types.iter().enumerate() {
        by_type.entry(t.clone()).or_default().push(i);
    }
    for ((t1, t2), (sig, eps)) in ff.nbfix.iter() {
        let (g1, g2) = match (by_type.get(t1), by_type.get(t2)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let mut combos: Vec<(usize, usize)> = Vec::new();
        if t1 != t2 {
            for a in g1 {
                for b in g2 {
                    if a != b {
                        combos.push((*a, *b));
                    }
                }
            }
        } else {
            for x in 0..g1.len() {
                for y in (x + 1)..g1.len() {
                    combos.push((g1[x], g1[y]));
                }
            }
        }
        for (i, j) in combos {
            if qm_set.contains(&i) || qm_set.contains(&j) {
                continue;
            }
            let key = (i.min(j), i.max(j));
            if excl12.contains(&key) || excl13.contains(&key) {
                continue;
            }
            tables.set(i, j, charges[i] * charges[j], *sig, *eps);
        }
    }
    let ql: Vec<usize> = qm_set.iter().cloned().collect();
    for a in 0..ql.len() {
        for b in (a + 1)..ql.len() {
            tables.set(ql[a], ql[b], 0.0, 1.0, 0.0);
        }
    }

    let order: Vec<usize> = if keep_order {
        (0..n).collect()
    } else {
        let mut o: Vec<usize> = ql.clone();
        o.extend((0..n).filter(|i| !qm_set.contains(i)));
        o
    };
    let mut inv = vec![0usize; n];
    for (nw, old) in order.iter().enumerate() {
        inv[*old] = nw;
    }

    let box_nm = struct_.box_nm;
    let pbc = box_nm[0] > 0.0;
    let xml = system_xml(
        &struct_, &types, &finalq, &masses, &bonds, &ub, &angles, &periodic, &harmonic, &rb,
        &cmaps, &ff, &tables, &qm_set, &order, &inv, cutoff_nm, pbc, box_nm,
    );
    std::fs::write(out_xml, xml)?;
    if keep_order {
        return Ok(n);
    }

    let mut gro = String::from("REST auto-built from topology (QM atoms first)\n");
    let _ = writeln!(gro, "{:5}", n);
    for nw in 0..n {
        let old = order[nw];
        let p = struct_.positions[old];
        let _ = writeln!(
            gro,
            "{:5}{:<5}{:>5}{:5}{:8.3}{:8.3}{:8.3}",
            nw / 100000 + 1,
            &struct_.resnames[old][..struct_.resnames[old].len().min(5)],
            &struct_.elements[old][..struct_.elements[old].len().min(5)],
            nw + 1,
            p[0] / 10.0,
            p[1] / 10.0,
            p[2] / 10.0
        );
    }
    let _ = writeln!(gro, "{:10.5}{:10.5}{:10.5}", box_nm[0], box_nm[1], box_nm[2]);
    std::fs::write(out_gro, gro)?;

    let mut qx = String::new();
    let _ = writeln!(qx, "{}", qm_set.len());
    let _ = writeln!(qx, "REST auto QM region (append link H after these, in order)");
    for nw in 0..order.len().min(qm_set.len()) {
        let old = order[nw];
        let p = struct_.positions[old];
        let _ = writeln!(qx, "{:<2} {:15.8} {:15.8} {:15.8}", struct_.elements[old], p[0], p[1], p[2]);
    }
    let qm_path = Path::new(out_xml).parent().unwrap_or(Path::new(".")).join("auto_qm.xyz");
    std::fs::write(qm_path, qx)?;
    Ok(n)
}

fn make_tip3p() -> Molecule {
    Molecule {
        name: "SOL".into(),
        types: vec!["OT".into(), "HT".into(), "HT".into()],
        charges: vec![super::mm::TIP3P_Q_O, super::mm::TIP3P_Q_H, super::mm::TIP3P_Q_H],
        masses: vec![super::mm::MASS_O_AMU, super::mm::MASS_H_AMU, super::mm::MASS_H_AMU],
        resnames: vec!["SOL".into(); 3],
        atomnames: vec!["OW".into(), "HW1".into(), "HW2".into()],
        resids: vec![1; 3],
        bonds: vec![
            (0, 1, super::mm::TIP3P_R_OH_NM, super::mm::TIP3P_K_OH_KJMOL_NM2),
            (0, 2, super::mm::TIP3P_R_OH_NM, super::mm::TIP3P_K_OH_KJMOL_NM2),
        ],
        angles: vec![(
            1,
            0,
            2,
            super::mm::TIP3P_THETA_HOH_DEG,
            super::mm::TIP3P_K_HOH_KJMOL_RAD2,
            0.0,
            0.0,
        )],
        exclusions: vec![(0, 1), (0, 2), (1, 2)],
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn system_xml(
    _st: &Structure,
    types: &[String],
    finalq: &[f64],
    masses: &[f64],
    bonds: &[(usize, usize, f64, f64)],
    ub: &[(usize, usize, f64, f64)],
    angles: &[(usize, usize, usize, f64, f64, f64, f64)],
    periodic: &[(usize, usize, usize, usize, f64, f64, u32)],
    harmonic: &[(usize, usize, usize, usize, f64, f64)],
    rb: &[(usize, usize, usize, usize, [f64; 6])],
    cmaps: &[(usize, usize, usize, usize, usize, CmapKey)],
    ff: &ForceField,
    tables: &ExceptionTable,
    qm: &std::collections::BTreeSet<usize>,
    order: &[usize],
    inv: &[usize],
    cutoff_nm: f64,
    pbc: bool,
    box_nm: [f64; 3],
) -> String {
    let n = order.len();
    let mut s = String::from("<?xml version=\"1.0\" ?>\n<System type=\"System\" version=\"1\">\n");
    let _ = writeln!(
        s,
        " <PeriodicBoxVectors><A x=\"{}\" y=\"0\" z=\"0\"/><B x=\"0\" y=\"{}\" z=\"0\"/><C x=\"0\" y=\"0\" z=\"{}\"/></PeriodicBoxVectors>",
        box_nm[0].max(0.1), box_nm[1].max(0.1), box_nm[2].max(0.1)
    );
    s.push_str(" <Particles>\n");
    for nw in 0..n {
        let _ = writeln!(s, "  <Particle mass=\"{}\"/>", masses[order[nw]]);
    }
    s.push_str(" </Particles>\n <Constraints/>\n <Forces>\n");

    let method = if pbc { 4 } else { 0 };
    let cutoff = if pbc { cutoff_nm.min(0.99 * 0.5 * box_nm.iter().cloned().fold(f64::MAX, f64::min)) } else { cutoff_nm };
    let _ = writeln!(
        s,
        "  <Force alpha=\"0\" cutoff=\"{cutoff}\" dispersionCorrection=\"1\" ewaldTolerance=\"1e-05\" exceptionsUsePeriodic=\"0\" forceGroup=\"0\" includeDirectSpace=\"1\" ljAlpha=\"0\" ljnx=\"0\" ljny=\"0\" ljnz=\"0\" method=\"{method}\" name=\"NonbondedForce\" nx=\"0\" ny=\"0\" nz=\"0\" recipForceGroup=\"-1\" rfDielectric=\"78.3\" switchingDistance=\"-1\" type=\"NonbondedForce\" useSwitchingFunction=\"0\" version=\"4\">"
    );
    s.push_str("   <GlobalParameters/>\n   <ParticleOffsets/>\n   <ExceptionOffsets/>\n   <Particles>\n");
    for nw in 0..n {
        let old = order[nw];
        let (sig, eps) = ff.atomtypes.get(&types[old]).cloned().unwrap_or((0.0, 0.0));
        let _ = writeln!(s, "    <Particle q=\"{}\" sig=\"{}\" eps=\"{}\"/>", finalq[old], sig, eps);
    }
    s.push_str("   </Particles>\n   <Exceptions>\n");
    for k in tables.keys.iter() {
        let (q, sig, eps) = tables.vals[k];
        let _ = writeln!(s, "    <Exception p1=\"{}\" p2=\"{}\" q=\"{}\" sig=\"{}\" eps=\"{}\"/>", inv[k.0], inv[k.1], q, sig, eps);
    }
    s.push_str("   </Exceptions>\n  </Force>\n");

    if !bonds.is_empty() || !ub.is_empty() {
        s.push_str("  <Force forceGroup=\"0\" name=\"HarmonicBondForce\" type=\"HarmonicBondForce\" usesPeriodic=\"0\" version=\"2\">\n   <Bonds>\n");
        for (a, b, r, k) in bonds {
            if qm.contains(a) && qm.contains(b) {
                continue;
            }
            let _ = writeln!(s, "    <Bond d=\"{r}\" k=\"{k}\" p1=\"{}\" p2=\"{}\"/>", inv[*a], inv[*b]);
        }
        for (a, c, r, k) in ub {
            if qm.contains(a) && qm.contains(c) {
                continue;
            }
            let _ = writeln!(s, "    <Bond d=\"{r}\" k=\"{k}\" p1=\"{}\" p2=\"{}\"/>", inv[*a], inv[*c]);
        }
        s.push_str("   </Bonds>\n  </Force>\n");
    }
    if !angles.is_empty() {
        s.push_str("  <Force forceGroup=\"0\" name=\"HarmonicAngleForce\" type=\"HarmonicAngleForce\" usesPeriodic=\"0\" version=\"2\">\n   <Angles>\n");
        for (a, b, c, th, k, _, _) in angles {
            if qm.contains(a) && qm.contains(b) && qm.contains(c) {
                continue;
            }
            let _ = writeln!(s, "    <Angle a=\"{}\" k=\"{k}\" p1=\"{}\" p2=\"{}\" p3=\"{}\"/>", th * DEG, inv[*a], inv[*b], inv[*c]);
        }
        s.push_str("   </Angles>\n  </Force>\n");
    }
    if !periodic.is_empty() {
        s.push_str("  <Force forceGroup=\"0\" name=\"PeriodicTorsionForce\" type=\"PeriodicTorsionForce\" usesPeriodic=\"0\" version=\"2\">\n   <Torsions>\n");
        for (a, b, c, d, phase, k, mlt) in periodic {
            if qm.contains(a) && qm.contains(b) && qm.contains(c) && qm.contains(d) {
                continue;
            }
            let _ = writeln!(s, "    <Torsion k=\"{k}\" p1=\"{}\" p2=\"{}\" p3=\"{}\" p4=\"{}\" periodicity=\"{mlt}\" phase=\"{phase}\"/>", inv[*a], inv[*b], inv[*c], inv[*d]);
        }
        s.push_str("   </Torsions>\n  </Force>\n");
    }
    if !harmonic.is_empty() {
        let energy = format!(
            "0.5*k*(thetap-theta0)^2; thetap = step(-(theta-theta0+pi))*2*pi+theta+step(theta-theta0-pi)*(-2*pi); pi = {}",
            std::f64::consts::PI
        );
        let _ = writeln!(s, "  <Force energy=\"{energy}\" forceGroup=\"0\" name=\"CustomTorsionForce\" type=\"CustomTorsionForce\" usesPeriodic=\"0\" version=\"3\">");
        s.push_str("   <PerTorsionParameters><Parameter name=\"theta0\"/><Parameter name=\"k\"/></PerTorsionParameters>\n   <GlobalParameters/>\n   <EnergyParameterDerivatives/>\n   <Torsions>\n");
        for (a, b, c, d, phi0, k) in harmonic {
            if qm.contains(a) && qm.contains(b) && qm.contains(c) && qm.contains(d) {
                continue;
            }
            let _ = writeln!(s, "    <Torsion p1=\"{}\" p2=\"{}\" p3=\"{}\" p4=\"{}\" param1=\"{phi0}\" param2=\"{k}\"/>", inv[*a], inv[*b], inv[*c], inv[*d]);
        }
        s.push_str("   </Torsions>\n  </Force>\n");
    }
    if !rb.is_empty() {
        s.push_str("  <Force forceGroup=\"0\" name=\"RBTorsionForce\" type=\"RBTorsionForce\" usesPeriodic=\"0\" version=\"2\">\n   <Torsions>\n");
        for (a, b, c, d, cs) in rb {
            if qm.contains(a) && qm.contains(b) && qm.contains(c) && qm.contains(d) {
                continue;
            }
            let _ = writeln!(
                s,
                "    <Torsion c0=\"{}\" c1=\"{}\" c2=\"{}\" c3=\"{}\" c4=\"{}\" c5=\"{}\" p1=\"{}\" p2=\"{}\" p3=\"{}\" p4=\"{}\"/>",
                cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], inv[*a], inv[*b], inv[*c], inv[*d]
            );
        }
        s.push_str("   </Torsions>\n  </Force>\n");
    }
    if !cmaps.is_empty() {
        let mut map_ids: BTreeMap<CmapKey, usize> = BTreeMap::new();
        let mut map_list: Vec<(usize, Vec<f64>)> = Vec::new();
        s.push_str("  <Force forceGroup=\"0\" name=\"CMAPTorsionForce\" type=\"CMAPTorsionForce\" usesPeriodic=\"0\" version=\"2\">\n   <Maps>\n");
        for (_, _, _, _, _, key) in cmaps {
            if map_ids.contains_key(key) {
                continue;
            }
            let grid = match ff.cmaptypes.get(key) {
                Some(g) => g,
                None => continue,
            };
            let nn = (grid.len() as f64).sqrt() as usize;
            if nn * nn != grid.len() {
                continue;
            }
            let mut transf = vec![0.0f64; nn * nn];
            for i in 0..nn {
                for j in 0..nn {
                    transf[i * nn + j] = grid[((j + nn / 2) % nn) * nn + ((i + nn / 2) % nn)];
                }
            }
            let id = map_ids.len();
            map_ids.insert(key.clone(), id);
            map_list.push((nn, transf));
            let _ = writeln!(s, "   <Map size=\"{nn}\">");
            for v in &map_list[id].1 {
                let _ = writeln!(s, "    <Energy e=\"{v}\"/>");
            }
            s.push_str("   </Map>\n");
        }
        s.push_str("   </Maps>\n   <Torsions>\n");
        for (a, b, c, d, e, key) in cmaps {
            if qm.contains(a) && qm.contains(b) && qm.contains(c) && qm.contains(d) && qm.contains(e) {
                continue;
            }
            if let Some(id) = map_ids.get(key) {
                let _ = writeln!(s, "    <Torsion a1=\"{}\" a2=\"{}\" a3=\"{}\" a4=\"{}\" b1=\"{}\" b2=\"{}\" b3=\"{}\" b4=\"{}\" map=\"{id}\"/>", inv[*a], inv[*b], inv[*c], inv[*d], inv[*b], inv[*c], inv[*d], inv[*e]);
            }
        }
        s.push_str("   </Torsions>\n  </Force>\n");
    }
    s.push_str(" </Forces>\n</System>\n");
    s
}
