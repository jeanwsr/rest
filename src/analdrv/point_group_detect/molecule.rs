//! The `Molecule` detection core. Port of `psi4/driver/qcdb/libmintsmolecule.py`
//! (`center_of_mass`, `inertia_tensor`/`molecule.py`, `rotor_type`,
//! `is_linear_planar`, `has_inversion`, `is_plane`, `is_axis`, `like_world_axis`,
//! `symmetry_frame`, `find_highest_point_group`, `set_full_point_group`) and
//! `molecule.py::rotational_symmetry_number`.

use super::bits;
use super::elements;
use super::geom::{atom_at_position, atom_present_in_geom, matrix_3d_rotation, matrix_3d_rotation_cn};
use super::linalg::diagonalize3x3symmat;
use super::matrix::{points_matmul, Matrix3};
use super::vec3::{self, Vec3, ZERO};

pub const DEFAULT_SYM_TOL: f64 = 1.0e-8;
pub const FULL_PG_TOL: f64 = 1.0e-8;
pub const NOISY_ZERO: f64 = 1.0e-8;

/// Full point-group template (the "n" is substituted later).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tmpl {
    Atom,
    DinfH,
    CinfV,
    Td,
    Oh,
    Ih,
    C1,
    Cs,
    Ci,
    Cn,
    Cnv,
    Cnh,
    Dn,
    Dnd,
    Dnh,
    Sn,
}

impl Tmpl {
    /// Schoenflies symbol with `n` substituted. Reference: `get_full_point_group`.
    pub fn full_name(self, n: u32) -> String {
        use Tmpl::*;
        match self {
            Atom => "ATOM".into(),
            DinfH => "D_inf_h".into(),
            CinfV => "C_inf_v".into(),
            Td => "Td".into(),
            Oh => "Oh".into(),
            Ih => "Ih".into(),
            C1 => "C1".into(),
            Cs => "Cs".into(),
            Ci => "Ci".into(),
            Cn => format!("C{}", n),
            Cnv => format!("C{}v", n),
            Cnh => format!("C{}h", n),
            Dn => format!("D{}", n),
            Dnd => format!("D{}d", n),
            Dnh => format!("D{}h", n),
            Sn => format!("S{}", n),
        }
    }

    /// Rotational symmetry number σ. Reference: `rotational_symmetry_number`.
    pub fn sigma(self, n: u32) -> f64 {
        use Tmpl::*;
        match self {
            Atom | C1 | Ci | Cs | CinfV => 1.0,
            DinfH => 2.0,
            Td => 12.0,
            Oh => 24.0,
            Ih => 60.0,
            Cn | Cnv | Cnh => n as f64,
            Dn | Dnd | Dnh => 2.0 * n as f64,
            Sn => n as f64 / 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rotor {
    Atom,
    Linear,
    Spherical,
    Prolate,
    Oblate,
    Asymmetric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisLike {
    X,
    Y,
    Z,
}

/// A molecule for point-group detection. Geometry is in **Bohr**.
pub struct SymmMolecule {
    pub symbols: Vec<String>,
    z: Vec<u8>,
    pub geom: Vec<[f64; 3]>,
    masses: Vec<f64>,
    pub tol: f64,
}

impl SymmMolecule {
    pub fn new(symbols: Vec<String>, geom: Vec<[f64; 3]>) -> Self {
        assert_eq!(symbols.len(), geom.len(), "symbols and geom length mismatch");
        let z: Vec<u8> = symbols.iter().map(|s| elements::symbol_to_z(s).unwrap_or(0)).collect();
        let masses = z.iter().map(|&z| elements::z_to_mass(z)).collect();
        Self { symbols, z, geom, masses, tol: DEFAULT_SYM_TOL }
    }

    /// Override per-atom masses (e.g. isotopic substitution to break symmetry).
    /// Length must equal the atom count.
    pub fn with_masses(mut self, masses: Vec<f64>) -> Self {
        assert_eq!(masses.len(), self.z.len(), "masses length mismatch");
        self.masses = masses;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn natom(&self) -> usize {
        self.z.len()
    }

    /// Reference `is_equivalent_to`: same Z and same mass (exact). Ghost status
    /// is irrelevant here (no ghosts in the test set).
    fn equiv(&self, i: usize, j: usize) -> bool {
        self.z[i] == self.z[j] && self.masses[i] == self.masses[j]
    }

    /// Mass-weighted center of mass. Reference: `center_of_mass`.
    fn center_of_mass(&self, g: &[[f64; 3]]) -> Vec3 {
        let mut ret = [0.0; 3];
        let mut total = 0.0;
        for i in 0..g.len() {
            let m = self.masses[i];
            ret = vec3::add(&ret, &vec3::scale(&g[i], m));
            total += m;
        }
        vec3::scale(&ret, 1.0 / total)
    }

    /// Mass-weighted inertia tensor. Reference: `inertia_tensor_partial`.
    fn inertia_tensor(&self, g: &[[f64; 3]]) -> Matrix3 {
        let mut t = [[0.0; 3]; 3];
        for i in 0..g.len() {
            let (x, y, z) = (g[i][0], g[i][1], g[i][2]);
            let m = self.masses[i];
            t[0][0] += m * (y * y + z * z);
            t[1][1] += m * (x * x + z * z);
            t[2][2] += m * (x * x + y * y);
            t[0][1] -= m * x * y;
            t[0][2] -= m * x * z;
            t[1][2] -= m * y * z;
        }
        t[1][0] = t[0][1];
        t[2][0] = t[0][2];
        t[2][1] = t[1][2];
        for r in 0..3 {
            for c in 0..3 {
                if t[r][c].abs() < ZERO {
                    t[r][c] = 0.0;
                }
            }
        }
        t
    }

    /// Reference: installed `rotor_type` (1.10.2).
    fn rotor_type(&self, g: &[[f64; 3]]) -> Rotor {
        let (mut evals, _evecs) = diagonalize3x3symmat(&self.inertia_tensor(g));
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rot_const: [f64; 3] = [
            if evals[0] > 1.0e-6 { 1.0 / evals[0] } else { 0.0 },
            if evals[1] > 1.0e-6 { 1.0 / evals[1] } else { 0.0 },
            if evals[2] > 1.0e-6 { 1.0 / evals[2] } else { 0.0 },
        ];

        let mut degen = 0;
        for i in 0..2 {
            for j in (i + 1)..3 {
                if degen >= 2 {
                    continue;
                }
                let rabs = (rot_const[i] - rot_const[j]).abs();
                let tmp = rot_const[i].max(rot_const[j]);
                let rel = if rabs > ZERO { rabs / tmp } else { 0.0 };
                if rel < self.tol {
                    degen += 1;
                }
            }
        }

        if self.natom() == 1 {
            Rotor::Atom
        } else if rot_const[0] == 0.0 {
            Rotor::Linear
        } else if degen == 2 {
            Rotor::Spherical
        } else if degen == 1 {
            if (rot_const[1] - rot_const[2]) < 1.0e-6 {
                Rotor::Prolate
            } else if (rot_const[0] - rot_const[1]) < 1.0e-6 {
                Rotor::Oblate
            } else {
                Rotor::Asymmetric
            }
        } else {
            Rotor::Asymmetric
        }
    }

    /// Reference: `is_linear_planar`.
    fn is_linear_planar(&self, g: &[[f64; 3]], tol: f64) -> (bool, bool) {
        if self.natom() < 3 {
            return (true, true);
        }
        let a = g[0];
        let b = g[1];
        let ba = vec3::normalize(&vec3::sub(&b, &a));
        let mut ca: Vec3 = [0.0; 3];
        let mut min_ba_dot_ca = 1.0;
        for i in 2..self.natom() {
            let tmp = vec3::normalize(&vec3::sub(&g[i], &a));
            if vec3::dot(&ba, &tmp).abs() < min_ba_dot_ca {
                ca = tmp;
                min_ba_dot_ca = vec3::dot(&ba, &tmp).abs();
            }
        }
        if min_ba_dot_ca >= 1.0 - tol {
            return (true, true);
        }
        let linear = false;
        if self.natom() < 4 {
            return (linear, true);
        }
        let baxca = vec3::normalize(&vec3::cross(&ba, &ca));
        for i in 2..self.natom() {
            let tmp = vec3::sub(&g[i], &a);
            if vec3::dot(&tmp, &baxca).abs() > tol {
                return (linear, false);
            }
        }
        (linear, true)
    }

    /// Reference: `has_inversion`.
    fn has_inversion(&self, g: &[[f64; 3]], origin: &Vec3, tol: f64) -> bool {
        for at in 0..self.natom() {
            let inverted = vec3::sub(&vec3::scale(origin, 2.0), &g[at]);
            match atom_at_position(g, &inverted, tol) {
                Some(idx) if self.equiv(idx, at) => {},
                _ => return false,
            }
        }
        true
    }

    /// Reference: `is_plane` (reflection through plane with normal `uperp` at `origin`).
    fn is_plane(&self, g: &[[f64; 3]], origin: &Vec3, uperp: &Vec3, tol: f64) -> bool {
        for i in 0..self.natom() {
            let a = vec3::sub(&g[i], origin);
            let apar = vec3::scale(uperp, vec3::dot(uperp, &a));
            let aperp = vec3::sub(&a, &apar);
            let reflected = vec3::add(&vec3::sub(&aperp, &apar), origin);
            match atom_at_position(g, &reflected, tol) {
                Some(idx) if self.equiv(idx, i) => {},
                _ => return false,
            }
        }
        true
    }

    /// Reference: `is_axis`.
    fn is_axis(&self, g: &[[f64; 3]], origin: &Vec3, axis: &Vec3, order: usize, tol: f64) -> bool {
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..self.natom() {
            let a = vec3::sub(&g[i], origin);
            for j in 1..order {
                let r = vec3::rotate(&a, j as f64 * two_pi / order as f64, axis);
                let r = vec3::add(&r, origin);
                match atom_at_position(g, &r, tol) {
                    Some(idx) if self.equiv(idx, i) => {},
                    _ => return false,
                }
            }
        }
        true
    }

    /// Is the vertical plane containing the z-axis at azimuth `theta` a mirror
    /// plane? Reflection across it maps `(x,y,z) -> (x cos2θ + y sin2θ,
    /// x sin2θ - y cos2θ, z)`. Every atom must map to an equivalent atom.
    /// (Replaces the reference's single-pivot reflect-x test, which depended on
    /// a favorable orientation — see `has_vertical_mirror`.)
    fn is_vertical_plane(&self, g: &[[f64; 3]], theta: f64, tol: f64) -> bool {
        let c2 = (2.0 * theta).cos();
        let s2 = (2.0 * theta).sin();
        for i in 0..self.natom() {
            let (x, y, z) = (g[i][0], g[i][1], g[i][2]);
            let refl = [x * c2 + y * s2, x * s2 - y * c2, z];
            match atom_at_position(g, &refl, tol) {
                Some(j) if self.equiv(j, i) => {},
                _ => return false,
            }
        }
        true
    }

    /// Robust σv detection: scan candidate vertical mirror planes and return
    /// true if any is a mirror. Fixes the orientation dependence of the
    /// reference's pivot approach (an `acos` sign ambiguity caused skew-rotated
    /// C3v molecules to lose their σv and report C3).
    ///
    /// A vertical plane at azimuth θ reflects an atom's xy-angle α to 2θ−α, so
    /// the only θ values that can be mirrors are: θ = αᵢ (an atom lying on the
    /// plane) and θ = (αᵢ+αⱼ)/2 (a mirror pair bisected by the plane), for
    /// equivalent atoms at equal distance from the z-axis. Planes are periodic
    /// mod π, so candidates are normalized to [0, π).
    fn has_vertical_mirror(&self, g: &[[f64; 3]], tol: f64) -> bool {
        let pi = std::f64::consts::PI;
        // off-axis atoms: (azimuth, radius², index)
        let off: Vec<(f64, f64, usize)> = (0..self.natom())
            .filter_map(|i| {
                let r2 = g[i][0] * g[i][0] + g[i][1] * g[i][1];
                if r2 > tol * tol {
                    Some((g[i][1].atan2(g[i][0]), r2, i))
                } else {
                    None
                }
            })
            .collect();
        if off.is_empty() {
            return false; // no off-axis atoms: no vertical mirror (linear case handled elsewhere)
        }

        let mut cands: Vec<f64> = Vec::new();
        for a in 0..off.len() {
            let (ang_a, r2_a, ia) = off[a];
            cands.push(ang_a); // on-plane candidate (atom lies on its own σv)
            for b in (a + 1)..off.len() {
                let (ang_b, r2_b, ib) = off[b];
                if !self.equiv(ia, ib) {
                    continue;
                }
                // mirror images are equidistant from the z-axis
                if (r2_a - r2_b).abs() > 1.0e-6 {
                    continue;
                }
                cands.push(0.5 * (ang_a + ang_b)); // bisector candidate
            }
        }

        // dedupe mod π
        let mut uniq: Vec<f64> = Vec::new();
        for c in cands {
            let c = c.rem_euclid(pi);
            let mut dup = false;
            for u in &uniq {
                let d = (c - u).abs().min((c - u + pi).abs()).min((c - u - pi).abs());
                if d < 1.0e-7 {
                    dup = true;
                    break;
                }
            }
            if !dup {
                uniq.push(c);
            }
        }

        for theta in uniq {
            if self.is_vertical_plane(g, theta, tol) {
                return true;
            }
        }
        false
    }

    /// Robust rotation-axis detection: does the molecule have a `order`-fold
    /// rotation axis (in any direction) through `origin`? Scans candidate axes
    /// derived from the atom positions: each atom direction `normalize(Aᵢ)`
    /// (axes through opposite atoms, e.g. C4 of an octahedron or C5 of an
    /// icosahedron) and each equivalent-pair bisector `normalize(Aᵢ+Aⱼ)` (axes
    /// through edge midpoints). Used to distinguish Oh (has C4) from Ih (no C4)
    /// independent of orientation — the reference tested S4 about the z-axis
    /// only, which depended on `symmetry_frame` aligning a C4 axis to z.
    fn has_rotation_axis(&self, g: &[[f64; 3]], origin: &Vec3, order: usize, tol: f64) -> bool {
        let positions: Vec<Vec3> = (0..self.natom()).map(|i| vec3::sub(&g[i], origin)).collect();
        // candidate axes: atom directions
        for i in 0..self.natom() {
            let a = positions[i];
            let na = vec3::dot(&a, &a);
            if na < tol * tol {
                continue;
            }
            let axis = vec3::normalize(&a);
            if self.is_axis(g, origin, &axis, order, tol) {
                return true;
            }
        }
        // candidate axes: equivalent-pair bisectors (axes through edge midpoints)
        for i in 0..self.natom() {
            let a = positions[i];
            let adota = vec3::dot(&a, &a);
            for j in 0..i {
                if !self.equiv(i, j) {
                    continue;
                }
                let b = positions[j];
                if (adota - vec3::dot(&b, &b)).abs() > 1.0e-6 {
                    continue;
                }
                let axis = vec3::add(&a, &b);
                if vec3::norm(&axis) < 1.0e-12 {
                    continue;
                }
                let axis = vec3::normalize(&axis);
                if self.is_axis(g, origin, &axis, order, tol) {
                    return true;
                }
            }
        }
        false
    }

    /// Reference: `like_world_axis`.
    fn like_world_axis(axis: &Vec3) -> (AxisLike, Vec3) {
        let worldx = [1.0, 0.0, 0.0];
        let worldy = [0.0, 1.0, 0.0];
        let worldz = [0.0, 0.0, 1.0];
        let xlike = vec3::dot(axis, &worldx).abs();
        let ylike = vec3::dot(axis, &worldy).abs();
        let zlike = vec3::dot(axis, &worldz).abs();
        if (xlike - ylike) > 1.0e-12 && (xlike - zlike) > 1.0e-12 {
            let a = if vec3::dot(axis, &worldx) < 0.0 { vec3::scale(axis, -1.0) } else { *axis };
            (AxisLike::X, a)
        } else if (ylike - zlike) > 1.0e-12 {
            let a = if vec3::dot(axis, &worldy) < 0.0 { vec3::scale(axis, -1.0) } else { *axis };
            (AxisLike::Y, a)
        } else {
            let a = if vec3::dot(axis, &worldz) < 0.0 { vec3::scale(axis, -1.0) } else { *axis };
            (AxisLike::Z, a)
        }
    }

    /// Tier A: highest D2h subgroup as a bit field. Reference:
    /// `find_highest_point_group`.
    fn find_highest_point_group(&self, g: &[[f64; 3]], tol: f64) -> u8 {
        let mut pg_bits = 0u8;
        for (op_bit, diag) in bits::tested_ops() {
            let mut found = true;
            for at in 0..self.natom() {
                let pos = vec3::naivemult(&g[at], &diag);
                match atom_at_position(g, &pos, tol) {
                    Some(idx) if self.equiv(idx, at) => {},
                    _ => {
                        found = false;
                        break;
                    },
                }
            }
            if found {
                pg_bits |= op_bit;
            }
        }
        pg_bits
    }

    /// Reference: `symmetry_frame`. Returns the 3×3 frame (columns = xaxis,
    /// yaxis, zaxis) that reorients the molecule so its symmetry axes align
    /// with the world axes.
    fn symmetry_frame(&self, g: &[[f64; 3]], tol: f64) -> Matrix3 {
        let com = self.center_of_mass(g);
        let shifted: Vec<[f64; 3]> = g.iter().map(|p| vec3::sub(p, &com)).collect();
        let worldx = [1.0, 0.0, 0.0];
        let worldy = [0.0, 1.0, 0.0];
        let worldz = [0.0, 0.0, 1.0];

        let mut sigma = [0.0; 3];
        let mut sigmav = [0.0; 3];
        let mut c2axis = [0.0; 3];
        let mut c2axisperp = [0.0; 3];

        let (linear, planar) = self.is_linear_planar(g, tol);
        let have_inversion = self.has_inversion(g, &com, tol);

        // --- check for C2 axis ---
        let mut have_c2axis = false;
        if self.natom() < 2 {
            have_c2axis = true;
            c2axis = [0.0, 0.0, 1.0];
        } else if linear {
            have_c2axis = true;
            c2axis = vec3::normalize(&vec3::sub(&g[1], &g[0]));
        } else if planar && have_inversion {
            let ba = vec3::normalize(&vec3::sub(&g[1], &g[0]));
            for i in 2..self.natom() {
                let ca = vec3::normalize(&vec3::sub(&g[i], &g[0]));
                let baxca = vec3::cross(&ba, &ca);
                if vec3::norm(&baxca) > tol {
                    have_c2axis = true;
                    c2axis = vec3::normalize(&baxca);
                    break;
                }
            }
        } else {
            'outer: for i in 0..self.natom() {
                let a = shifted[i];
                let adota = vec3::dot(&a, &a);
                for j in 0..=i {
                    if !self.equiv(i, j) {
                        continue;
                    }
                    let b = shifted[j];
                    if (adota - vec3::dot(&b, &b)).abs() > tol {
                        continue;
                    }
                    let axis = vec3::add(&a, &b);
                    if vec3::norm(&axis) < tol {
                        continue;
                    }
                    let axis = vec3::normalize(&axis);
                    if self.is_axis(g, &com, &axis, 2, tol) {
                        have_c2axis = true;
                        c2axis = axis;
                        break 'outer;
                    }
                }
            }
        }

        let mut c2like = AxisLike::Z;
        if have_c2axis {
            let (like, axis) = Self::like_world_axis(&c2axis);
            c2like = like;
            c2axis = axis;
        }

        // --- check for C2 axis perpendicular to first ---
        let mut have_c2axisperp = false;
        if have_c2axis {
            if self.natom() < 2 {
                have_c2axisperp = true;
                c2axisperp = [1.0, 0.0, 0.0];
            } else if linear {
                if have_inversion {
                    have_c2axisperp = true;
                    c2axisperp = vec3::perp_unit(&c2axis, &[0.0, 0.0, 1.0]);
                }
            } else {
                'outer: for i in 0..self.natom() {
                    let a = vec3::sub(&g[i], &com);
                    let adota = vec3::dot(&a, &a);
                    for j in 0..i {
                        if !self.equiv(i, j) {
                            continue;
                        }
                        let b = vec3::sub(&g[j], &com);
                        if (adota - vec3::dot(&b, &b)).abs() > tol {
                            continue;
                        }
                        let axis = vec3::add(&a, &b);
                        if vec3::norm(&axis) < tol {
                            continue;
                        }
                        let axis = vec3::normalize(&axis);
                        if vec3::dot(&axis, &c2axis).abs() > tol {
                            continue;
                        }
                        if self.is_axis(g, &com, &axis, 2, tol) {
                            have_c2axisperp = true;
                            c2axisperp = axis;
                            break 'outer;
                        }
                    }
                }
            }
        }

        if have_c2axisperp {
            let (mut c2perplike, snapped) = Self::like_world_axis(&c2axisperp);
            c2axisperp = snapped;
            // try to make c2axis the z axis
            if c2perplike == AxisLike::Z {
                std::mem::swap(&mut c2axisperp, &mut c2axis);
                c2perplike = c2like;
                c2like = AxisLike::Z;
            }
            if c2like != AxisLike::Z {
                c2axis = if c2like == AxisLike::X {
                    vec3::cross(&c2axis, &c2axisperp)
                } else {
                    vec3::cross(&c2axisperp, &c2axis)
                };
                let (like, axis) = Self::like_world_axis(&c2axis);
                c2like = like;
                c2axis = axis;
            }
            if c2perplike == AxisLike::Y {
                c2axisperp = vec3::cross(&c2axisperp, &c2axis);
                let (_like, axis) = Self::like_world_axis(&c2axisperp);
                c2axisperp = axis;
            }
        }

        // --- check for vertical plane ---
        let mut have_sigmav = false;
        if have_c2axis {
            if self.natom() < 2 {
                have_sigmav = true;
                sigmav = c2axisperp;
            } else if linear {
                have_sigmav = true;
                if have_c2axisperp {
                    sigmav = c2axisperp;
                } else {
                    sigmav = vec3::perp_unit(&c2axis, &[0.0, 0.0, 1.0]);
                }
            } else {
                'outer: for i in 0..self.natom() {
                    let a = vec3::sub(&g[i], &com);
                    let adota = vec3::dot(&a, &a);
                    for j in 0..=i {
                        if !self.equiv(i, j) {
                            continue;
                        }
                        let b = vec3::sub(&g[j], &com);
                        if (adota - vec3::dot(&b, &b)).abs() > tol {
                            continue;
                        }
                        let inplane = vec3::add(&b, &a);
                        let norm_inplane = vec3::norm(&inplane);
                        if norm_inplane < tol {
                            continue;
                        }
                        let inplane = vec3::scale(&inplane, 1.0 / norm_inplane);
                        let perp = vec3::cross(&c2axis, &inplane);
                        let norm_perp = vec3::norm(&perp);
                        if norm_perp < tol {
                            continue;
                        }
                        let perp = vec3::scale(&perp, 1.0 / norm_perp);
                        if self.is_plane(g, &com, &perp, tol) {
                            have_sigmav = true;
                            sigmav = perp;
                            break 'outer;
                        }
                    }
                }
            }
        }

        if have_sigmav {
            let (sigmavlike, axis) = Self::like_world_axis(&sigmav);
            sigmav = axis;
            if c2like == AxisLike::Z && sigmavlike == AxisLike::Y {
                sigmav = vec3::cross(&sigmav, &c2axis);
            } else if c2like == AxisLike::Y && sigmavlike == AxisLike::Z {
                sigmav = vec3::cross(&c2axis, &sigmav);
            }
        }

        // --- check for any sigma plane (only if no inversion and no c2 axis) ---
        let mut have_sigma = false;
        if !have_inversion && !have_c2axis {
            if planar {
                let ba = vec3::normalize(&vec3::sub(&g[1], &g[0]));
                for i in 2..self.natom() {
                    let ca = vec3::normalize(&vec3::sub(&g[i], &g[0]));
                    let baxca = vec3::cross(&ba, &ca);
                    if vec3::norm(&baxca) > tol {
                        have_sigma = true;
                        sigma = vec3::normalize(&baxca);
                        break;
                    }
                }
            } else {
                'outer: for i in 0..self.natom() {
                    let a = vec3::sub(&g[i], &com);
                    let adota = vec3::dot(&a, &a);
                    for j in 0..i {
                        if !self.equiv(i, j) {
                            continue;
                        }
                        let b = vec3::sub(&g[j], &com);
                        let bdotb = vec3::dot(&b, &b);
                        if (adota - bdotb).abs() > tol {
                            continue;
                        }
                        let perp = vec3::sub(&b, &a);
                        let norm_perp = vec3::norm(&perp);
                        if norm_perp < tol {
                            continue;
                        }
                        let perp = vec3::scale(&perp, 1.0 / norm_perp);
                        if self.is_plane(g, &com, &perp, tol) {
                            have_sigma = true;
                            sigma = perp;
                            break 'outer;
                        }
                    }
                }
            }
        }

        if have_sigma {
            let xlikeness = vec3::dot(&sigma, &worldx).abs();
            let ylikeness = vec3::dot(&sigma, &worldy).abs();
            let zlikeness = vec3::dot(&sigma, &worldz).abs();
            if xlikeness > ylikeness && xlikeness > zlikeness {
                if vec3::dot(&sigma, &worldx) < 0.0 {
                    sigma = vec3::scale(&sigma, -1.0);
                }
            } else if ylikeness > zlikeness {
                if vec3::dot(&sigma, &worldy) < 0.0 {
                    sigma = vec3::scale(&sigma, -1.0);
                }
            } else if vec3::dot(&sigma, &worldz) < 0.0 {
                sigma = vec3::scale(&sigma, -1.0);
            }
        }

        // --- assemble the three axes ---
        let mut xaxis = worldx;
        let mut zaxis = worldz;
        if have_c2axis {
            zaxis = c2axis;
            if have_sigmav {
                xaxis = sigmav;
            } else if have_c2axisperp {
                xaxis = c2axisperp;
            } else {
                xaxis = vec3::perp_unit(&zaxis, &zaxis);
            }
        } else if have_sigma {
            zaxis = sigma;
            xaxis = vec3::perp_unit(&zaxis, &zaxis);
        }

        for k in 0..3 {
            if zaxis[k].abs() < NOISY_ZERO {
                zaxis[k] = 0.0;
            }
            if xaxis[k].abs() < NOISY_ZERO {
                xaxis[k] = 0.0;
            }
        }
        let yaxis = vec3::scale(&vec3::cross(&xaxis, &zaxis), -1.0);

        let mut frame = super::matrix::zero();
        for i in 0..3 {
            frame[i][0] = xaxis[i];
            frame[i][1] = yaxis[i];
            frame[i][2] = zaxis[i];
        }
        frame
    }

    /// Tier B: determine the full point group template and `n`. Reference:
    /// `set_full_point_group`. `g` must already be COM-recentered and
    /// symmetry-frame rotated (the state of `self.geometry()` after
    /// `update_geometry`).
    fn set_full_point_group(&self, g: &[[f64; 3]], tol: f64) -> (Tmpl, u32) {
        // local recentered copy (g is already COM-recentered; com ~ 0)
        let com = self.center_of_mass(g);
        let mut geom: Vec<[f64; 3]> = g.iter().map(|p| vec3::sub(p, &com)).collect();

        let rotor = self.rotor_type(g);
        let pg_bits = self.find_highest_point_group(g, tol);
        let d2h_subgroup = bits::bits_to_basic_name(pg_bits);
        let op_i = self.has_inversion(g, &[0.0, 0.0, 0.0], tol);

        let z_axis = [0.0, 0.0, 1.0];

        match rotor {
            Rotor::Atom => (Tmpl::Atom, 0),
            Rotor::Linear => {
                if op_i {
                    (Tmpl::DinfH, 0)
                } else {
                    (Tmpl::CinfV, 0)
                }
            },
            Rotor::Spherical => {
                if !op_i {
                    (Tmpl::Td, 3)
                } else {
                    // Oh has a C4 axis (anywhere); Ih does not (its highest is
                    // C5). Test for a C4 axis in any direction rather than S4
                    // about z only — the latter depended on `symmetry_frame`
                    // aligning a C4 axis to z and mis-reported Oh as Ih for
                    // generically-oriented molecules.
                    let origin = [0.0, 0.0, 0.0];
                    if self.has_rotation_axis(g, &origin, 4, tol) {
                        (Tmpl::Oh, 4)
                    } else {
                        (Tmpl::Ih, 5)
                    }
                }
            },
            Rotor::Asymmetric => {
                let (tmpl, n) = match d2h_subgroup {
                    "c1" => (Tmpl::C1, 1u32),
                    "ci" => (Tmpl::Ci, 1),
                    "c2" => (Tmpl::Cn, 2),
                    "cs" => (Tmpl::Cs, 1),
                    "d2" => (Tmpl::Dn, 2),
                    "c2v" => (Tmpl::Cnv, 2),
                    "c2h" => (Tmpl::Cnh, 2),
                    "d2h" => (Tmpl::Dnh, 2),
                    _ => (Tmpl::C1, 1),
                };
                (tmpl, n)
            },
            Rotor::Prolate | Rotor::Oblate => {
                // symmetric top
                let (evals, evecs) = diagonalize3x3symmat(&self.inertia_tensor(g));
                // zip eigenvalues with eigenvectors (columns of evecs -> rows), sort ascending
                let mut ev_list: Vec<(f64, Vec3)> =
                    (0..3).map(|i| (evals[i], [evecs[0][i], evecs[1][i], evecs[2][i]])).collect();
                ev_list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let i_evals: [f64; 3] = [ev_list[0].0, ev_list[1].0, ev_list[2].0];
                let i_evecs: [Vec3; 3] = [ev_list[0].1, ev_list[1].1, ev_list[2].1];

                let mut unique_axis = 1usize;
                if (i_evals[0] - i_evals[1]).abs() < tol {
                    unique_axis = 2;
                } else if (i_evals[1] - i_evals[2]).abs() < tol {
                    unique_axis = 0;
                }
                let old_axis = i_evecs[unique_axis];

                let ddot = vec3::dot(&z_axis, &old_axis);
                let phi = if (ddot - 1.0).abs() < 1.0e-10 {
                    0.0
                } else if (ddot + 1.0).abs() < 1.0e-10 {
                    std::f64::consts::PI
                } else {
                    ddot.acos()
                };

                if phi.abs() > 1.0e-14 {
                    let rot_axis = vec3::cross(&z_axis, &old_axis);
                    geom = matrix_3d_rotation(&geom, &rot_axis, phi, false);
                }

                let cn_z = matrix_3d_rotation_cn(&geom, &z_axis, false, tol, 0);
                let sn_z = matrix_3d_rotation_cn(&geom, &z_axis, true, tol, 0);

                // sigma_h (xy plane): reflect z
                let mut op_sigma_h = true;
                for at in 0..self.natom() {
                    if geom[at][2].abs() < tol {
                        continue;
                    }
                    let test_atom = [geom[at][0], geom[at][1], -geom[at][2]];
                    if !atom_present_in_geom(&geom, &test_atom, tol) {
                        op_sigma_h = false;
                        break;
                    }
                }

                // sigma_v: robust scan of candidate vertical mirror planes.
                // (Reference pivoted one atom into the yz plane and tested
                // reflect-x; that depended on the pivot's xy-angle sign and
                // failed for generically-oriented Cnv molecules such as NH3.)
                let op_sigma_v = self.has_vertical_mirror(&geom, tol);

                // perpendicular C2's (pair by Z only, per reference). This scan
                // is invariant under rotations about z, so it needs no pivot.
                let mut is_d = false;
                'pair: for i in 0..self.natom() {
                    let a = geom[i];
                    let adota = vec3::dot(&a, &a);
                    for j in 0..i {
                        if self.z[i] != self.z[j] {
                            continue;
                        }
                        let b = geom[j];
                        if (adota - vec3::dot(&b, &b)).abs() > 1.0e-6 {
                            continue;
                        }
                        let axis = vec3::add(&a, &b);
                        if vec3::norm(&axis) < 1.0e-12 {
                            continue;
                        }
                        let axis = vec3::normalize(&axis);
                        if vec3::dot(&axis, &z_axis).abs() > 1.0e-6 {
                            continue;
                        }
                        if matrix_3d_rotation_cn(&geom, &axis, false, tol, 2) == 2 {
                            is_d = true;
                            break 'pair;
                        }
                    }
                }

                let cn = cn_z as u32;
                let sn = sn_z as u32;
                if sn == 2 * cn && !is_d {
                    return (Tmpl::Sn, sn);
                }
                if is_d {
                    if op_sigma_h && op_sigma_v {
                        return (Tmpl::Dnh, cn);
                    } else if sn == 2 * cn {
                        return (Tmpl::Dnd, cn);
                    } else {
                        return (Tmpl::Dn, cn);
                    }
                } else {
                    if op_sigma_h && sn == cn {
                        return (Tmpl::Cnh, cn);
                    } else if op_sigma_v {
                        return (Tmpl::Cnv, cn);
                    } else {
                        return (Tmpl::Cn, cn);
                    }
                }
            },
        }
    }

    /// Run the full detection pipeline. Returns `(template, n)`.
    pub(crate) fn detect_inner(&self) -> (Tmpl, u32) {
        let tol = self.tol;
        // 1. COM recenter
        let com = self.center_of_mass(&self.geom);
        let mut g: Vec<[f64; 3]> = self.geom.iter().map(|p| vec3::sub(p, &com)).collect();
        // 2. symmetry_frame reorientation
        let frame = self.symmetry_frame(&g, tol);
        g = points_matmul(&g, &frame);
        // 3. Tier A (inside set_full_point_group) + Tier B
        self.set_full_point_group(&g, tol)
    }
}
