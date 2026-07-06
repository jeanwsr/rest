//! Molecular point-group detection and rotational symmetry number.
//!
//! Self-contained, geometry-based detector. The key observation used here:
//! the rotational symmetry number **sigma equals the number of distinct
//! proper-rotation symmetry operations** (Cn^k, including the identity E).
//! Improper operations (inversion, mirrors, Sn) do not contribute to sigma,
//! so we only need to find all Cn axes and count the operations they generate.
//!
//! Reference approach: classical symmetry-element search (cf. Beruski &
//! Vidal, J. Comput. Chem. 2013; V. Beruski's algorithms). Candidate rotation
//! axes are drawn from atom positions, inter-atomic directions, and the
//! principal inertia axes; each candidate is tested as a Cn (n=2..8) axis.

use rest_tensors::matrix::matrix_blas_lapack::_dsyev;
use tensors::MatrixFull;

type V3 = [f64; 3];

/// Default position-matching tolerance (Bohr). Optimized symmetric geometries
/// typically deviate from ideal symmetry by << 1e-2 Bohr.
const POS_TOL: f64 = 1.0e-2;
/// Direction parallelism tolerance for deduplicating candidate axes.
const DIR_TOL: f64 = 1.0e-3;

#[derive(Debug, Clone)]
pub struct PointGroupInfo {
    /// Best-effort Schoenflies label, e.g. "Td", "C2v", "D6h", "D∞h".
    pub label: String,
    /// Rotational symmetry number = number of proper rotations in the group.
    pub sigma: u32,
    /// Highest proper rotation order found (principal axis order; 1 if none).
    pub principal_order: u32,
    pub is_linear: bool,
    pub has_inversion: bool,
}

/// Detect the point group and rotational symmetry number from a molecular
/// geometry. Positions are expected in **Bohr** as a `[3, natm]` matrix
/// (`position[[coord, atom]]`); `elems` gives element symbols.
pub fn detect_point_group(position: &MatrixFull<f64>, elems: &[String]) -> PointGroupInfo {
    detect_point_group_tol(position, elems, POS_TOL)
}

pub fn detect_point_group_tol(position: &MatrixFull<f64>, elems: &[String], tol: f64) -> PointGroupInfo {
    let natm = elems.len();
    let raw: Vec<V3> = (0..natm)
        .map(|a| [position[[0, a]], position[[1, a]], position[[2, a]]])
        .collect();
    let cen = centroid(&raw);
    let pos: Vec<V3> = raw.iter().map(|p| sub(*p, cen)).collect();

    if natm == 1 {
        return PointGroupInfo {
            label: "C1".into(), sigma: 1, principal_order: 1,
            is_linear: false, has_inversion: false,
        };
    }

    // unit-mass inertia tensor (centered) for linearity test + candidate axes
    let (eigvals, eigvecs) = inertia_eig(&pos);
    let is_linear = eigvals[0] < 1.0e-6 * eigvals[1].max(1.0e-12);

    // inversion center?
    let inv_op = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    let has_inv = is_symmetry(&inv_op, &pos, elems, tol);

    if is_linear {
        let (label, sigma) = if has_inv { ("D∞h", 2u32) } else { ("C∞v", 1u32) };
        return PointGroupInfo {
            label: label.into(), sigma, principal_order: 0,
            is_linear: true, has_inversion: has_inv,
        };
    }

    // ---- candidate axis directions ----
    let mut dirs: Vec<V3> = Vec::new();
    for i in 0..natm {
        if let Some(u) = normalize(pos[i]) { dirs.push(u); }
    }
    for i in 0..natm {
        for j in (i + 1)..natm {
            for &s in &[1.0, -1.0] {
                let d = [
                    pos[i][0] + s * pos[j][0],
                    pos[i][1] + s * pos[j][1],
                    pos[i][2] + s * pos[j][2],
                ];
                if let Some(u) = normalize(d) { dirs.push(u); }
            }
        }
    }
    for k in 0..3 {
        dirs.push([eigvecs[[0, k]], eigvecs[[1, k]], eigvecs[[2, k]]]);
    }
    let dirs = dedupe_dirs(dirs, 1.0 - 1.0e-4);

    // ---- find Cn axes (record maximal order per physical axis) ----
    let orders = [8u32, 6, 5, 4, 3, 2];
    let mut found: Vec<(V3, u32)> = Vec::new();
    for d in &dirs {
        for &n in &orders {
            let op = rot_matrix(*d, 2.0 * std::f64::consts::PI / n as f64);
            if is_symmetry(&op, &pos, elems, tol) {
                found.push((*d, n));
                break; // highest order for this candidate direction
            }
        }
    }
    let axes = cluster_axes(found, 1.0 - DIR_TOL);

    // ---- detect mirror planes (only for labeling) ----
    let mut mirror_normals: Vec<V3> = Vec::new();
    for d in &dirs {
        let op = refl_matrix(*d);
        if is_symmetry(&op, &pos, elems, tol) { mirror_normals.push(*d); }
    }
    let mirror_normals = dedupe_dirs(mirror_normals, 1.0 - DIR_TOL);

    // principal axis = highest-order axis
    let (prin_dir, pn) = axes
        .iter()
        .max_by_key(|(_, o)| *o)
        .copied()
        .unwrap_or(([1.0, 0.0, 0.0], 1u32));

    // ---- sigma = number of distinct proper rotations (incl. identity) ----
    let mut ops: Vec<[f64; 9]> = Vec::new();
    ops.push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]); // identity
    for (d, n) in &axes {
        for k in 1..*n {
            ops.push(rot_matrix(*d, 2.0 * std::f64::consts::PI * k as f64 / *n as f64));
        }
    }
    let sigma = dedupe_ops(&ops) as u32;

    let label = classify(&axes, &mirror_normals, has_inv, prin_dir, pn);

    PointGroupInfo {
        label, sigma, principal_order: pn, is_linear: false, has_inversion: has_inv,
    }
}

// ── linear-algebra / geometry helpers ──────────────────────────────────

#[inline] fn centroid(p: &[V3]) -> V3 {
    let mut c = [0.0; 3];
    for q in p { for k in 0..3 { c[k] += q[k]; } }
    for k in 0..3 { c[k] /= p.len() as f64; }
    c
}
#[inline] fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
#[inline] fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
#[inline] fn vnorm(a: V3) -> f64 { (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() }
#[inline] fn normalize(a: V3) -> Option<V3> {
    let n = vnorm(a);
    if n < 1.0e-10 { None } else { Some([a[0] / n, a[1] / n, a[2] / n]) }
}

fn inertia_eig(pos: &[V3]) -> (Vec<f64>, MatrixFull<f64>) {
    let mut ixx = 0.0; let mut iyy = 0.0; let mut izz = 0.0;
    let mut ixy = 0.0; let mut ixz = 0.0; let mut iyz = 0.0;
    for p in pos {
        ixx += p[1] * p[1] + p[2] * p[2];
        iyy += p[0] * p[0] + p[2] * p[2];
        izz += p[0] * p[0] + p[1] * p[1];
        ixy -= p[0] * p[1];
        ixz -= p[0] * p[2];
        iyz -= p[1] * p[2];
    }
    let data = vec![ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz];
    let m = MatrixFull::from_vec([3, 3], data).unwrap();
    let (vo, vals, _) = _dsyev(&m, 'V');
    let vecs = vo.unwrap_or_else(|| MatrixFull::new([3, 3], 0.0));
    (vals, vecs)
}

/// Row-major 3x3 rotation matrix about unit axis `u` by angle `theta` (Rodrigues).
fn rot_matrix(u: V3, theta: f64) -> [f64; 9] {
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;
    let (x, y, z) = (u[0], u[1], u[2]);
    [
        t * x * x + c,     t * x * y - s * z, t * x * z + s * y,
        t * x * y + s * z, t * y * y + c,     t * y * z - s * x,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c,
    ]
}

/// Row-major reflection matrix through the plane with unit normal `n`: R = I - 2 n n^T.
fn refl_matrix(n: V3) -> [f64; 9] {
    let (x, y, z) = (n[0], n[1], n[2]);
    [
        1.0 - 2.0 * x * x, -2.0 * x * y,      -2.0 * x * z,
        -2.0 * y * x,      1.0 - 2.0 * y * y, -2.0 * y * z,
        -2.0 * z * x,      -2.0 * z * y,      1.0 - 2.0 * z * z,
    ]
}

#[inline] fn apply(m: &[f64; 9], p: V3) -> V3 {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
        m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
        m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
    ]
}

/// Test whether `op` is a symmetry operation: every atom must map (within tol)
/// onto an unused atom of the same element (bijective matching).
fn is_symmetry(op: &[f64; 9], pos: &[V3], elems: &[String], tol: f64) -> bool {
    let n = pos.len();
    let mut used = vec![false; n];
    for i in 0..n {
        let gp = apply(op, pos[i]);
        let mut best: Option<(usize, f64)> = None;
        for j in 0..n {
            if used[j] || elems[j] != elems[i] { continue; }
            let dx = gp[0] - pos[j][0];
            let dy = gp[1] - pos[j][1];
            let dz = gp[2] - pos[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < tol && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        match best {
            Some((j, _)) => used[j] = true,
            None => return false,
        }
    }
    true
}

/// Deduplicate directions, treating `d` and `-d` as the same axis.
fn dedupe_dirs(dirs: Vec<V3>, thr: f64) -> Vec<V3> {
    let mut out: Vec<V3> = Vec::new();
    for d in dirs {
        let mut dup = false;
        for u in &out {
            if dot(d, *u).abs() > thr { dup = true; break; }
        }
        if !dup { out.push(d); }
    }
    out
}

/// Merge axes that are (anti)parallel, keeping the maximal order per cluster.
fn cluster_axes(axes: Vec<(V3, u32)>, thr: f64) -> Vec<(V3, u32)> {
    let mut out: Vec<(V3, u32)> = Vec::new();
    for (d, n) in axes {
        let mut merged = false;
        for e in out.iter_mut() {
            if dot(d, e.0).abs() > thr {
                if n > e.1 { *e = (d, n); }
                merged = true;
                break;
            }
        }
        if !merged { out.push((d, n)); }
    }
    out
}

/// Count distinct 3x3 operation matrices (within 1e-6 elementwise).
fn dedupe_ops(ops: &[[f64; 9]]) -> usize {
    let mut uniq: Vec<[f64; 9]> = Vec::new();
    for op in ops {
        let mut dup = false;
        for u in &uniq {
            let mut same = true;
            for i in 0..9 {
                if (op[i] - u[i]).abs() > 1.0e-6 { same = false; break; }
            }
            if same { dup = true; break; }
        }
        if !dup { uniq.push(*op); }
    }
    uniq.len()
}

/// Best-effort Schoenflies label from the axis/mirror inventory. sigma is
/// always correct; the label is informational and may be coarse for rare
/// (e.g. pure Sn) groups.
fn classify(
    axes: &[(V3, u32)],
    mirrors: &[V3],
    has_inv: bool,
    prin: V3,
    pn: u32,
) -> String {
    let (mut n2, mut n3, mut n4, mut n5) = (0u32, 0u32, 0u32, 0u32);
    for (_, o) in axes {
        match *o { 2 => n2 += 1, 3 => n3 += 1, 4 => n4 += 1, 5 => n5 += 1, _ => {} }
    }
    if n5 >= 6 { return if has_inv { "Ih" } else { "I" }.into(); }
    if n4 >= 3 { return if has_inv { "Oh" } else { "O" }.into(); }
    if n3 >= 4 {
        if !mirrors.is_empty() { return "Td".into(); }
        if has_inv { return "Th".into(); }
        return "T".into();
    }
    if pn <= 1 {
        return if has_inv { "Ci" }
            else if !mirrors.is_empty() { "Cs" }
            else { "C1" }.into();
    }
    let perp_c2 = axes.iter().filter(|(d, o)| *o == 2 && dot(*d, prin).abs() < 0.9).count();
    let has_sh = mirrors.iter().any(|n| dot(*n, prin).abs() > 0.9);
    let has_sv = mirrors.iter().any(|n| dot(*n, prin).abs() < 0.9);
    if perp_c2 >= 1 {
        if has_sh { format!("D{}h", pn) }
        else if has_sv { format!("D{}d", pn) }
        else { format!("D{}", pn) }
    } else if has_sh {
        format!("C{}h", pn)
    } else if has_sv {
        format!("C{}v", pn)
    } else {
        format!("C{}", pn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_pos(flat: &[f64]) -> MatrixFull<f64> {
        let natm = flat.len() / 3;
        let mut m = MatrixFull::new([3, natm], 0.0);
        for a in 0..natm {
            for c in 0..3 { m[[c, a]] = flat[a * 3 + c]; }
        }
        m
    }

    #[test]
    fn ch4_is_td_sigma12() {
        let pos = mk_pos(&[
            0.0, 0.0, 0.0,
            1.0897, 1.0897, 1.0897,
            -1.0897, -1.0897, 1.0897,
            1.0897, -1.0897, -1.0897,
            -1.0897, 1.0897, -1.0897,
        ]);
        let elems: Vec<String> = vec!["C".into(), "H".into(), "H".into(), "H".into(), "H".into()];
        let pg = detect_point_group(&pos, &elems);
        assert_eq!(pg.sigma, 12);
        assert_eq!(pg.label, "Td");
    }

    #[test]
    fn h2o_is_c2v_sigma2() {
        let pos = mk_pos(&[
            0.0, 0.0, 0.0,
            0.0, 0.757, 0.587,
            0.0, -0.757, 0.587,
        ]);
        let elems: Vec<String> = vec!["O".into(), "H".into(), "H".into()];
        let pg = detect_point_group(&pos, &elems);
        assert_eq!(pg.sigma, 2);
        assert_eq!(pg.label, "C2v");
    }

    #[test]
    fn co2_is_dinfh_sigma2() {
        let pos = mk_pos(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.16, 0.0, 0.0, -1.16]);
        let elems: Vec<String> = vec!["C".into(), "O".into(), "O".into()];
        let pg = detect_point_group(&pos, &elems);
        assert_eq!(pg.sigma, 2);
        assert!(pg.is_linear);
        assert_eq!(pg.label, "D∞h");
    }
}
