use std::f64::consts;
use rstsr::prelude::*;
use serde::{Deserialize, Serialize};
use tensors::{MatrixFull, MatrixUpper, BasicMatrix};
use super::SurfaceSwitchType;
use crate::constants::solvent as data;
use crate::dft::gen_grids::angular_grid;
use crate::geom_io::{GeomCell, get_mass_charge};


//=============================================================================
//  Surface Related Structures
//=============================================================================

/// Container for surface calculation data
#[derive(Clone)]
pub struct SurfaceCalc{
    pub grid_coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub switch_fun: Vec<f64>,
    pub norm_vec: Vec<(f64, f64, f64)>,
    pub charge_exp: Vec<f64>,
   // pub charge: Vec<f64>,
    pub R_vdw: Vec<f64>,
    pub area: Vec<f64>,
}

impl SurfaceCalc{
    pub fn new() -> Self{
        SurfaceCalc{
            grid_coords: vec![],
            weights: vec![],
            switch_fun: vec![],
            norm_vec: vec![],
            charge_exp: vec![],
            R_vdw: vec![],
            area: vec![],
        }
    }
}

/// surface configurations
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceVdwGaussianCfg {
    pub lebedev_degree: usize,
    pub atom_radii: Option<Vec<f64>>,
    pub vdw_scale: f64,
    pub switch_type: SurfaceSwitchType,
}

impl Default for SurfaceVdwGaussianCfg {
    fn default() -> Self {
        SurfaceVdwGaussianCfg { lebedev_degree: 302, atom_radii: None, vdw_scale: 1.2, switch_type: SurfaceSwitchType::default() }
    }
}
//lebedev_degree is usually 302

/// Gaussian VDW surface representation
#[non_exhaustive]
#[derive(Clone)]
pub struct SurfaceVdwGaussian {
    pub cfg: SurfaceVdwGaussianCfg,
    pub atomic_num: Vec<usize>,
    pub atom_coords: MatrixFull<f64>,
    pub surface_calc: SurfaceCalc,
    pub gslice_by_atom: Vec<(usize,usize)>,

}

impl SurfaceVdwGaussian {
    pub fn new(cfg: SurfaceVdwGaussianCfg, geom: &GeomCell) -> Self {
        let mut surface_calc = SurfaceCalc::new();
        let gslice_by_atom = vec![];
        let mass_charge = get_mass_charge(&geom.elem);
        let atomic_num = mass_charge.iter().map(|&(_, charge)| charge as usize).collect();
        let atom_coords = geom.position.clone();
        SurfaceVdwGaussian { cfg, atomic_num, atom_coords, surface_calc, gslice_by_atom }
    }

    /// PySCF's `solvent.pcm.gen_surface`
    pub fn build(&mut self) {
        // unwrap atom_radii or calculate from atom_charges
        let atom_radii = self.cfg.atom_radii.clone().unwrap_or_else(|| {
            self.atomic_num.iter()
                .map(|&z| if z == 1 { 2.0786987370215684 * self.cfg.vdw_scale } else { data::VDW_RADII[z] * self.cfg.vdw_scale })
                .collect()
        });

        // sanity check
        let natm = self.atomic_num.len();
        if atom_radii.len() != natm {
            panic!("Number of atom radii must match number of atom charges.");
        }
        if self.atom_coords.size[1] != natm {
            panic!("Number of atom coordinates must match number of atom charges.");
        }

        // basic quantities
        //let device = DeviceBLAS::default();
        let lebedev_degree = self.cfg.lebedev_degree;
        let unit_sphere = angular_grid(lebedev_degree);
        let unit_quads = unit_sphere.0;
        let unit_weights = unit_sphere.1;

        let R_J = atom_radii;

        let R_sw_J: Vec<f64> = R_J.iter().map(|&r| r * (14.0 / (lebedev_degree as f64)).sqrt()).collect();
        let alpha_J: Vec<f64> = R_J.iter().zip(R_sw_J.iter())
            .map(|(&r_vdw, &r_sw)| 0.5 + r_vdw/r_sw - ((r_vdw/r_sw).powi(2) - 1.0/28.0).sqrt())
            .collect();
        let R_in_J: Vec<f64> = R_J.iter().zip(alpha_J.iter()).zip(R_sw_J.iter())
            .map(|((&r_vdw, &alpha), &r_sw)| r_vdw - alpha * r_sw)
            .collect();


        let mut p = 0;
        for ia in 0..natm {
            let r_vdw = R_J[ia];

            // preparation for switch function
            // the grid points for atom ia, which is the unit sphere scaled by r_vdw and shifted by atom_coords
            let atom_grid: Vec<[f64; 3]> = unit_quads.iter()
                .map(|&(x,y,z)| [
                    r_vdw * x + self.atom_coords[(0, ia)],
                    r_vdw * y + self.atom_coords[(1, ia)],
                    r_vdw * z + self.atom_coords[(2, ia)]
                ]).collect();
            //let atom_grid = r_vdw * &unit_quads + atom_coords.i((ia, ..));

            let n_grids = atom_grid.len();
            // distance from the grid points of to all atoms, r_iJ, with shape (natm, n_grids)
            let mut r_iJ = MatrixFull::<f64>::new([natm, n_grids], 0.0);
            for j in 0..n_grids {
                for i in 0..natm {
                    r_iJ[(i, j)] = dist_point_atom(&atom_grid[j], &self.atom_coords, i);
                }
            }

            let w_i: Vec<f64> = unit_weights.iter().map(|&x| x * (4.0 * consts::PI)) .collect();
            //zeta_i = \frac{\zeta}{r_vdw \sqrt{w_i}}
            let zeta_i: Vec<f64> = w_i.iter().map(|&w| data::solvent_lebedev_scale_factor(lebedev_degree) / (r_vdw * w.sqrt())).collect();

            let mut d_iJ =MatrixFull::<f64>::new([natm,n_grids], 0.0);
            for j in 0..n_grids{
                for i in 0..natm{
                    if i == ia {
                        d_iJ[(i,j)] = 1.0;
                    }
                    else{
                        d_iJ[(i,j)] = (r_iJ[(i,j)] - R_in_J[i])/R_sw_J[i];

                    }
                }
            }
            // switch function
            //let f_iJ = match self.cfg.switch_type {
            //    SurfaceSwitchType::SWIG => SurfaceSwitchType::element_switch_swig(R_J.view(), r_iJ.view(), lebedev_degree),
            //    SurfaceSwitchType::ISWIG => SurfaceSwitchType::element_switch_iswig(R_J.view(), r_iJ.view(), zeta_i.view()),
            //};
            //let swf_i = SurfaceSwitchType::switch(f_iJ.view(), ia);
            let f_iJ = switch_matrix(d_iJ);
            // swf_i = numpy.prod(f_iJ, axis=0), means if a grid is in the cavity of any other atom, its switch function is zero
            let swf_i: Vec<f64> = f_iJ.iter_columns_full()
                .map(|col| col.iter().product())
                .collect();

            // zero out the switch function that is too small
            let idx: Vec<usize> = w_i.iter().zip(swf_i.iter())
                .map(|(&w, &s)| (w * s > 1e-16) as usize)
                .collect();
            let nidx = idx.iter().sum();
            //let nidx = idx.iter().filter(|&&x| x).count();
            let (p0, p1) = (p, p + nidx);

            let tmp_area:Vec<f64> = bool_select(&w_i, &idx).iter()
                .zip(bool_select(&swf_i, &idx).iter())
                .map(|(&w,&s)| r_vdw.powi(2) * w * s)
                .collect();
            self.gslice_by_atom.push((p0, p1));
            self.surface_calc.grid_coords.extend(bool_select(&atom_grid, &idx));
            self.surface_calc.weights.extend(bool_select(&w_i, &idx));
            self.surface_calc.switch_fun.extend(bool_select(&swf_i, &idx));
            self.surface_calc.norm_vec.extend(bool_select(&unit_quads, &idx));
            self.surface_calc.charge_exp.extend(bool_select(&zeta_i, &idx));
            self.surface_calc.R_vdw.extend(vec![r_vdw; nidx]);
            self.surface_calc.area.extend(tmp_area);

            p += nidx;
        }

    }

}


//-----------------------------------------------------------------------------
//  3D Coordinate utilities
//-----------------------------------------------------------------------------

/// Distance between two points given as [f64; 3]
#[inline]
pub fn dist_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Distance between a point [f64; 3] and an atom stored in a [3, natm] column-major matrix
#[inline]
pub fn dist_point_atom(point: &[f64; 3], atom_coords: &MatrixFull<f64>, ia: usize) -> f64 {
    let dx = point[0] - atom_coords[(0, ia)];
    let dy = point[1] - atom_coords[(1, ia)];
    let dz = point[2] - atom_coords[(2, ia)];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Dot product of vector (a - b) with normal n (all as [f64; 3])
#[inline]
pub fn diff_dot_normal(a: &[f64; 3], b: &[f64; 3], n: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * n[0] + dy * n[1] + dz * n[2]
}

//=============================================================================
//  Switch Functions
//=============================================================================

/// Utility function for boolean selection based on a mask
pub fn bool_select<T: Clone>(data: &[T], mask: &[usize]) -> Vec<T> {
    data.iter()
        .zip(mask.iter())
        .filter_map(|(d, &m)| if m != 0 { Some(d.clone()) } else { None })
        .collect()
}

/// switch function for a matrix of d_iJ
pub fn switch_matrix(d: MatrixFull<f64>) -> MatrixFull<f64> {
    let mut f = MatrixFull::<f64>::new(d.size, 0.0);
    for i in 0..d.size[0] {
        for j in 0..d.size[1] {
            if d[(i,j)] < 1e-8{
                f[(i,j)] = 0.0;
            }
            else{
                f[(i,j)] = switch_h(d[(i,j)]);
            }
        }
    }
    f
}

pub fn switch_h(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x.powi(3) * (10.0 - 15.0 * x + 6.0 * x.powi(2))
    }
}

//=============================================================================
//  Linear Algebra Utilities
//=============================================================================

/// Solve V·x = b where V is symmetric positive definite, using Cholesky decomposition.
/// Uses LAPACK's dpotrf (Cholesky) + forward/backward substitution.
pub fn solve_cholesky(v: &MatrixFull<f64>, b: &[f64]) -> Vec<f64> {
    let n = v.size[0];
    assert!(v.size[1] == n && b.len() == n, "solve_cholesky: dimension mismatch");

    // Copy V to mutable working matrix for in-place Cholesky
    let mut v_work = v.clone();

    // Cholesky factorization: V = L·L^T (L stored in lower triangle)
    v_work.to_matrixfullslicemut().lapack_dpotrf(b'L');

    let mut x = b.to_vec();

    // Forward substitution: L·y = b
    // L[i,j] = v_work[(i, j)] for j <= i (lower triangle)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += v_work[(i, j)] * x[j];
        }
        x[i] = (x[i] - sum) / v_work[(i, i)];
    }

    // Backward substitution: L^T·x = y
    // L^T[i,j] = L[j,i] = v_work[(j, i)]
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += v_work[(j, i)] * x[j];
        }
        x[i] = (x[i] - sum) / v_work[(i, i)];
    }

    x
}

/// Solve linear system Ax = b using LU decomposition without explicitly computing the inverse of A
pub fn solve_lu(a: &MatrixFull<f64>, a_ipiv: &Vec<i32>, b: &[f64]) -> Option<Vec<f64>>
{
    let n = a.size[0];
    // 2. 应用行置换到右端项 b
    let mut x = b.to_vec();
    for i in 0..n {
        if a_ipiv[i] as usize != i + 1 {
            x.swap(i, a_ipiv[i] as usize - 1);
        }
    }

    // 3. 前向替换求解 L·y = Pb (L 是单位下三角)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += a[(i, j)] * x[j];
        }
        x[i] = (x[i] - sum); // L[i,i] = 1.0
    }

    // 4. 后向替换求解 U·x = y
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += a[(i,j)] * x[j];
        }
        if a[(i,i)].abs() < 1e-15 {
            println!("Singular matrix detected at diagonal element {}", i);
            return None;
        }
        x[i] = (x[i] - sum) / a[(i,i)];
    }

    Some(x)
}

pub fn solve_lu_transpose(
    lu: &MatrixFull<f64>,   // 紧凑存储的 L 和 U
    ipiv: &[i32],           // 置换信息
    b: &[f64],              // 右端项
) -> Option<Vec<f64>> {
    let n = lu.size()[0];
    if b.len() != n { return None; }

    let mut x = b.to_vec();

    // 1. 解 U^T y = x (前向替换)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lu[(j, i)] * x[j];
        }
        if lu[(i, i)].abs() < 1e-15 { return None; }
        x[i] = (x[i] - sum) / lu[(i, i)];
    }

    // 2. 解 L^T z = y (后向替换，L^T 是单位上三角)
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += lu[(j, i)] * x[j];
        }
        x[i] -= sum;
    }

    // 3. 应用置换 P 得到最终解 x = P z
    // 注意：ipiv 记录的是分解时的行交换，正向应用即可
    for i in (0..n).rev() {   // 从后往前应用，避免覆盖问题
        let p = ipiv[i] as usize;
        if p != i + 1 {
            x.swap(i, p - 1);
        }
    }

    Some(x)
}
