#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BiasPotential {
    Cosine,
    Harmonic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CvType {
    Dihedral,
    Distance,
    Angle,
    DistanceDiff,
}

impl CvType {

    pub fn n_atoms(self) -> usize {
        match self {
            CvType::Dihedral | CvType::DistanceDiff => 4,
            CvType::Angle => 3,
            CvType::Distance => 2,
        }
    }

    pub fn angular(self) -> bool {
        matches!(self, CvType::Dihedral | CvType::Angle)
    }
}

#[derive(Debug, Clone)]
pub struct UmbrellaBias {
    pub cv: CvType,

    pub atoms: Vec<usize>,

    pub center: f64,

    pub kappa_hartree: f64,
    pub potential: BiasPotential,

    pub sum_center_ang: f64,
    pub sum_kappa_hartree: f64,
}

fn wrap_pi(angle: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut a = angle.rem_euclid(two_pi);
    if a > std::f64::consts::PI {
        a -= two_pi;
    }
    a
}

pub fn dihedral_rad(pos: &[f64], atoms: [usize; 4]) -> f64 {
    let p = |a: usize| [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
    let (pi, pj, pk, pl) = (p(atoms[0]), p(atoms[1]), p(atoms[2]), p(atoms[3]));
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let b1 = sub(pj, pi);
    let b2 = sub(pk, pj);
    let b3 = sub(pl, pk);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let m = cross(n1, n2);
    let b2_norm = dot(b2, b2).sqrt();
    let y = dot(m, b2) / b2_norm;
    let x = dot(n1, n2);
    y.atan2(x)
}

fn dihedral_gradient(pos: &[f64], atoms: [usize; 4]) -> [[f64; 3]; 4] {
    let p = |a: usize| [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
    let (pi, pj, pk, pl) = (p(atoms[0]), p(atoms[1]), p(atoms[2]), p(atoms[3]));
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let b1 = sub(pj, pi);
    let b2 = sub(pk, pj);
    let b3 = sub(pl, pk);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let m = cross(n1, n2);
    let b2_len = dot(b2, b2).sqrt();
    let b2_hat = [b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len];

    let x = dot(n1, n2);
    let y = dot(m, b2_hat);
    let r2 = x * x + y * y;

    let gx_n1 = n2;
    let gx_n2 = n1;
    let gy_n1 = cross(n2, b2_hat);
    let gy_n2 = cross(b2_hat, n1);
    let gy_b2h = m;

    let g_n1: [f64; 3] = std::array::from_fn(|c| (x * gy_n1[c] - y * gx_n1[c]) / r2);
    let g_n2: [f64; 3] = std::array::from_fn(|c| (x * gy_n2[c] - y * gx_n2[c]) / r2);
    let g_b2h: [f64; 3] = std::array::from_fn(|c| x * gy_b2h[c] / r2);

    let adj_b1 = cross(b2, g_n1);
    let mut adj_b2 = cross(g_n1, b1);
    let adj_b3 = cross(g_n2, b2);
    let t = cross(b3, g_n2);
    for c in 0..3 {
        adj_b2[c] += t[c];
        let tangential = g_b2h[c] - dot(g_b2h, b2_hat) * b2_hat[c];
        adj_b2[c] += tangential / b2_len;
    }

    let mut grad = [[0.0f64; 3]; 4];
    for c in 0..3 {
        grad[0][c] -= adj_b1[c];
        grad[1][c] += adj_b1[c];
        grad[1][c] -= adj_b2[c];
        grad[2][c] += adj_b2[c];
        grad[2][c] -= adj_b3[c];
        grad[3][c] += adj_b3[c];
    }
    grad
}

fn distance_val_grad(pos: &[f64], atoms: [usize; 2]) -> (f64, [[f64; 3]; 2]) {
    let p = |a: usize| [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
    let (pi, pj) = (p(atoms[0]), p(atoms[1]));
    let d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let dhat = [d[0] / r, d[1] / r, d[2] / r];

    (r, [[-dhat[0], -dhat[1], -dhat[2]], dhat])
}

fn angle_val_grad(pos: &[f64], atoms: [usize; 3]) -> (f64, [[f64; 3]; 3]) {
    let p = |a: usize| [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
    let (pi, pj, pk) = (p(atoms[0]), p(atoms[1]), p(atoms[2]));
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let u = sub(pi, pj);
    let v = sub(pk, pj);
    let un = dot(u, u).sqrt();
    let vn = dot(v, v).sqrt();
    let cos_t = (dot(u, v) / (un * vn)).clamp(-1.0, 1.0);
    let theta = cos_t.acos();
    let sin_t = (1.0 - cos_t * cos_t).sqrt().max(1.0e-30);
    if sin_t < 1.0e-12 {

        return (theta, [[0.0; 3]; 3]);
    }
    let uhat = [u[0] / un, u[1] / un, u[2] / un];
    let vhat = [v[0] / vn, v[1] / vn, v[2] / vn];

    let mut g_u = [0.0f64; 3];
    let mut g_v = [0.0f64; 3];
    for c in 0..3 {
        g_u[c] = (cos_t * uhat[c] - vhat[c]) / (un * sin_t);
        g_v[c] = (cos_t * vhat[c] - uhat[c]) / (vn * sin_t);
    }
    let mut g_mid = [0.0f64; 3];
    for c in 0..3 {
        g_mid[c] = -g_u[c] - g_v[c];
    }
    (theta, [g_u, g_mid, g_v])
}

impl UmbrellaBias {

    pub fn new(
        cv: CvType,
        atoms: &[usize],
        center: f64,
        kappa_kcalmol: f64,
        potential: BiasPotential,
        sum_center_ang: f64,
        sum_kappa_kcalmol: f64,
    ) -> anyhow::Result<Self> {
        if atoms.len() != cv.n_atoms() {
            return Err(anyhow::anyhow!(
                "this umbrella CV needs exactly {} atoms, got {:?}",
                cv.n_atoms(),
                atoms
            ));
        }
        if sum_kappa_kcalmol != 0.0 && cv != CvType::DistanceDiff {
            return Err(anyhow::anyhow!(
                "umbrella_sum_kappa is only supported for umbrella_cv = \"distance_diff\""
            ));
        }
        let kappa_hartree = kappa_kcalmol / crate::constants::HARTREE2KCALMOL;
        let center = if cv.angular() {
            center.to_radians()
        } else {
            center
        };
        Ok(UmbrellaBias {
            cv,
            atoms: atoms.to_vec(),
            center,
            kappa_hartree,
            potential,
            sum_center_ang,
            sum_kappa_hartree: sum_kappa_kcalmol / crate::constants::HARTREE2KCALMOL,
        })
    }

    fn sum_cv_and_gradient(&self, pos: &[f64]) -> Option<(f64, Vec<[f64; 3]>)> {
        if self.cv != CvType::DistanceDiff || self.sum_kappa_hartree == 0.0 {
            return None;
        }
        let a1 = [self.atoms[0], self.atoms[1]];
        let a2 = [self.atoms[2], self.atoms[3]];
        let (r1, g1) = distance_val_grad(pos, a1);
        let (r2, g2) = distance_val_grad(pos, a2);
        let bohr = crate::constants::BOHR;
        let g = vec![
            [bohr * g1[0][0], bohr * g1[0][1], bohr * g1[0][2]],
            [bohr * g1[1][0], bohr * g1[1][1], bohr * g1[1][2]],
            [bohr * g2[0][0], bohr * g2[0][1], bohr * g2[0][2]],
            [bohr * g2[1][0], bohr * g2[1][1], bohr * g2[1][2]],
        ];
        Some(((r1 + r2) * bohr, g))
    }

    fn cv_and_gradient(&self, pos: &[f64]) -> (f64, Vec<[f64; 3]>) {
        match self.cv {
            CvType::Dihedral => {
                let a = [
                    self.atoms[0],
                    self.atoms[1],
                    self.atoms[2],
                    self.atoms[3],
                ];
                (dihedral_rad(pos, a), dihedral_gradient(pos, a).to_vec())
            }
            CvType::Distance => {
                let a = [self.atoms[0], self.atoms[1]];
                let (r, g) = distance_val_grad(pos, a);

                let g_ang: Vec<[f64; 3]> = g
                    .iter()
                    .map(|row| std::array::from_fn(|c| row[c] * crate::constants::BOHR))
                    .collect();
                (r * crate::constants::BOHR, g_ang)
            }
            CvType::Angle => {
                let a = [self.atoms[0], self.atoms[1], self.atoms[2]];
                let (theta, g) = angle_val_grad(pos, a);
                (theta, g.to_vec())
            }
            CvType::DistanceDiff => {
                let a1 = [self.atoms[0], self.atoms[1]];
                let a2 = [self.atoms[2], self.atoms[3]];
                let (r1, g1) = distance_val_grad(pos, a1);
                let (r2, g2) = distance_val_grad(pos, a2);
                let bohr = crate::constants::BOHR;

                let g = [
                    [
                        bohr * g1[0][0],
                        bohr * g1[0][1],
                        bohr * g1[0][2],
                    ],
                    [
                        bohr * g1[1][0],
                        bohr * g1[1][1],
                        bohr * g1[1][2],
                    ],
                    [
                        -bohr * g2[0][0],
                        -bohr * g2[0][1],
                        -bohr * g2[0][2],
                    ],
                    [
                        -bohr * g2[1][0],
                        -bohr * g2[1][1],
                        -bohr * g2[1][2],
                    ],
                ];
                ((r1 - r2) * bohr, g.to_vec())
            }
        }
    }

    fn delta(&self, q: f64) -> f64 {
        if self.cv.angular() {
            wrap_pi(q - self.center)
        } else {
            q - self.center
        }
    }

    pub fn dvalue(&self, q: f64) -> f64 {
        let d = self.delta(q);
        match self.potential {
            BiasPotential::Cosine => self.kappa_hartree * d.sin(),
            BiasPotential::Harmonic => self.kappa_hartree * d,
        }
    }

    pub fn energy(&self, q: f64) -> f64 {
        let d = self.delta(q);
        match self.potential {
            BiasPotential::Cosine => self.kappa_hartree * (1.0 - d.cos()),
            BiasPotential::Harmonic => 0.5 * self.kappa_hartree * d * d,
        }
    }

    pub fn energy_and_forces(&self, pos: &[f64]) -> (f64, Vec<[f64; 3]>) {
        let (q, dqdp) = self.cv_and_gradient(pos);
        let mut energy = self.energy(q);
        let du = self.dvalue(q);
        let mut forces: Vec<[f64; 3]> = dqdp
            .iter()
            .map(|row| std::array::from_fn(|c| -du * row[c]))
            .collect();
        if let Some((s, dsdp)) = self.sum_cv_and_gradient(pos) {
            let d = s - self.sum_center_ang;
            energy += 0.5 * self.sum_kappa_hartree * d * d;
            let du_sum = self.sum_kappa_hartree * d;
            for (k, row) in dsdp.iter().enumerate() {
                for c in 0..3 {
                    forces[k][c] -= du_sum * row[c];
                }
            }
        }
        (energy, forces)
    }

    pub fn csv_columns(&self) -> Vec<&'static str> {
        match self.cv {
            CvType::Dihedral => vec!["phi_deg"],
            CvType::Angle => vec!["theta_deg"],
            CvType::Distance => vec!["cv_angstrom"],
            CvType::DistanceDiff => vec!["d1_A", "d2_A", "xi", "s"],
        }
    }

    pub fn display_values(&self, pos: &[f64]) -> Vec<f64> {
        match self.cv {
            CvType::DistanceDiff => {
                let (r1, _) = distance_val_grad(pos, [self.atoms[0], self.atoms[1]]);
                let (r2, _) = distance_val_grad(pos, [self.atoms[2], self.atoms[3]]);
                let bohr = crate::constants::BOHR;
                let (d1, d2) = (r1 * bohr, r2 * bohr);
                vec![d1, d2, d1 - d2, d1 + d2]
            }
            CvType::Dihedral | CvType::Angle => {
                let (q, _) = self.cv_and_gradient(pos);
                vec![q.to_degrees()]
            }
            CvType::Distance => {
                let (q, _) = self.cv_and_gradient(pos);
                vec![q]
            }
        }
    }
}
