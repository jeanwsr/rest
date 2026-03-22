use std::fmt;
use std::collections::HashMap;
use std::f64::consts;
use rest_libcint::gto::prelude_dev::X;
use rest_libcint::{CINTR2CDATA, CintType};
use rstsr_core::prelude_dev::shape;
use serde::{Deserialize, Serialize};
use tensors::{map_full_to_upper, map_upper_to_full, ri, BasicMatUp, BasicMatrix, MathMatrix, MatrixFull, MatrixFullSlice, MatrixFullSliceMut, MatrixUpper, MatrixUpperSlice};
//use rstsr as rt;
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dgemm_scaled};
use crate::molecule_io::Molecule;
use crate::tensors::{TensorOpt,TensorOptMut,TensorSlice};
use crate::geom_io::{GeomCell,MOrC, GeomUnit, get_mass_charge};
use crate::constants::solvent as data;
use crate::dft::gen_grids::angular_grid;


//=============================================================================
//  Configuration Structures
//=============================================================================

/// PCM method enumeration

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PcmMethod {
    #[default]
    CPCM,
    COSMO,
    IEFPCM,
    SSVPE,
}

impl fmt::Display for PcmMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcmMethod::COSMO => write!(f, "COSMO"),
            PcmMethod::CPCM => write!(f, "CPCM"),
            PcmMethod::IEFPCM => write!(f, "IEFPCM"),
            PcmMethod::SSVPE => write!(f, "SSVPE"),
        }
    }
}

impl<'d> Deserialize<'d> for PcmMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_uppercase().as_str() {
            "CPCM" => Ok(PcmMethod::CPCM),
            "COSMO" => Ok(PcmMethod::COSMO),
            "IEFPCM" => Ok(PcmMethod::IEFPCM),
            "SSVPE" | "SS(V)PE" => Ok(PcmMethod::SSVPE),
            _ => Err(serde::de::Error::custom(format!("Unknown PCM method: {}", s))),
        }
    }
}

/// PCM configuration
/// #[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PcmObjectCfg {
    pub method: PcmMethod,
    pub epsilon: f64,
}

impl PcmObjectCfg {
    pub fn build(method: PcmMethod, epsilon: f64) -> Self {
        PcmObjectCfg { method, epsilon }
    }
}

impl Default for PcmObjectCfg {
    fn default() -> Self {
        PcmObjectCfg { method: PcmMethod::default(), epsilon: 78.3553 }
    }
}

/// Surface switch function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum SurfaceSwitchType {
    SWIG,
    #[default]
    ISWIG,
}

impl<'d> Deserialize<'d> for SurfaceSwitchType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_uppercase().as_str() {
            "SWIG" => Ok(SurfaceSwitchType::SWIG),
            "ISWIG" => Ok(SurfaceSwitchType::ISWIG),
            _ => Err(serde::de::Error::custom(format!("Unknown switch type: {}", s))),
        }
    }
}

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

        let R_J = atom_radii.clone();

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
            let mut r_iJ =MatrixFull::<f64>::new([natm,n_grids], 0.0);
            for j in 0..n_grids {
                for i in 0..natm {
                    let dx = atom_grid[j][0] - self.atom_coords[(0, i)];
                    let dy = atom_grid[j][1] - self.atom_coords[(1, i)];
                    let dz = atom_grid[j][2] - self.atom_coords[(2, i)];
                    r_iJ[(i,j)] = (dx*dx + dy*dy + dz*dz).sqrt();
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
//  PCM Static Structures 
//=============================================================================

/// Static PCM data (computed once per calculation)
#[derive(Clone)]
pub struct PcmStatic {
    pub A: Vec<f64>,
    pub D: MatrixFull<f64>,
    pub S: MatrixFull<f64>,
    pub K: MatrixFull<f64>,
    pub R: MatrixFull<f64>,
    pub f_epsilon: f64,
    pub v_grids_n: Vec<f64>,
}

impl PcmStatic{
    pub fn get_A_D_S(surface: &SurfaceVdwGaussian) -> (Vec<f64>, MatrixFull<f64>, MatrixFull<f64>) {
        const PI: f64 = std::f64::consts::PI;

        let R_vdw = &surface.surface_calc.R_vdw;
        let switch_fun = &surface.surface_calc.switch_fun;
        let weights = &surface.surface_calc.weights;

        //The matrix A is diagonal and consists of the surface element areas, ai.
        let A: Vec<f64> = weights
        .iter().zip(R_vdw.iter()).zip(switch_fun.iter())
        .map(|((&w,&r),&s)| w * r.powi(2) * s).collect();

        let charge_exp = &surface.surface_calc.charge_exp;
        let grid_coords = &surface.surface_calc.grid_coords;
        let n_grids = grid_coords.len();
        let grid_capacity = n_grids * (n_grids + 1) /2;
        let norm_vec = &surface.surface_calc.norm_vec;

        //println!("DEBUG get_A_D_S: n_grids = {}", n_grids);
        //println!("DEBUG get_A_D_S: R_vdw len = {}", R_vdw.len());
        //println!("DEBUG get_A_D_S: switch_fun len = {}", switch_fun.len());
        //println!("DEBUG get_A_D_S: weights len = {}", weights.len());

        //println!("DEBUG get_A_D_S: grid_capacity = {}", grid_capacity);
        //assert!(n_grids == R_vdw.len() && n_grids == switch_fun.len() && n_grids == weights.len() && n_grids == charge_exp.len() 
        //&& n_grids == norm_vec.len(), "Length of grid-related vectors must match the number of grid points.");
        
        if n_grids == 0 {
            panic!("ERROR: No surface grid points generated!");
        }

        let mut zeta_ij = Vec::with_capacity(grid_capacity);
        for j in 0..n_grids{
            for i in 0..j+1{
                zeta_ij.push(charge_exp[i]*charge_exp[j]/(charge_exp[i].powi(2)+charge_exp[j].powi(2)).sqrt() );
            }
        }

        let mut r_ij = Vec::with_capacity(grid_capacity);
        for j in 0..n_grids{
            for i in 0..j+1{
                if i == j {
                    r_ij.push(1.0);
                }
                else{
                    let dx = grid_coords[i][0] - grid_coords[j][0];
                    let dy = grid_coords[i][1] - grid_coords[j][1];
                    let dz = grid_coords[i][2] - grid_coords[j][2];
                    let r = (dx*dx + dy*dy + dz*dz).sqrt();
                    r_ij.push(r);
                }
            }
        }
        //let r_ij = MatrixUpper::from_vec(grid_capacity, tmp_r_ij).unwrap();
        //let r_ij = r_ij.to_matrixfull();
        let zeta_r_ij: Vec<f64> = r_ij.iter()
            .zip(zeta_ij.iter()).map(|(&r, &z)| z*r).collect();
        
        // S_ij = erf(zeta_ij * r_ij) / r_ij, with S_ii = charge_exp[i] * (2/pi)^0.5 / switch_fun[i]
        let mut tmp_S_ij = zeta_r_ij.iter().zip(r_ij.iter())
            .map(|(&z, &r)| libm::erf(z) / r).collect();
        let mut S_ij: MatrixFull<f64> = MatrixUpper::from_vec(grid_capacity, tmp_S_ij).unwrap().to_matrixfull().unwrap();
        for i in 0..n_grids{
            let s_ii = charge_exp[i] * (2.0 / PI).sqrt() / switch_fun[i];
            S_ij[(i,i)] = s_ii;
        }

        let r_ij_matrix: MatrixFull<f64> = MatrixUpper::from_vec(grid_capacity, r_ij).unwrap().to_matrixfull().unwrap();
        let zeta_r_ij_matrix: MatrixFull<f64> = MatrixUpper::from_vec(grid_capacity, zeta_r_ij).unwrap().to_matrixfull().unwrap();
        // n_r_ij = \vec{n_j} * \vec{r_ij}, where \vec{n_j} is the normal vector at grid point j, and \vec{r_ij} is the vector from grid point j to grid point i.
        let mut n_r_ij: Vec<f64> = Vec::with_capacity(n_grids * n_grids);
        for j in 0..n_grids{
            for i in 0..n_grids{
                let dx = grid_coords[i][0] - grid_coords[j][0];
                let dy = grid_coords[i][1] - grid_coords[j][1];
                let dz = grid_coords[i][2] - grid_coords[j][2];
                n_r_ij.push( dx*norm_vec[j].0 + dy*norm_vec[j].1 + dz*norm_vec[j].2 );
            }
        }
    
        //r_scale = \vec{n_j} * \vec{r_ij}/ r_ij^3, 
        let r_scale: Vec<f64> = n_r_ij.iter().zip(r_ij_matrix.iter())
            .map(|(&n, &r)| n / r.powi(3)).collect();
        let f_zeta_r_ij: Vec<f64> = zeta_r_ij_matrix.iter()
            .map(|&x| libm::erf(x) - 2.0 / PI.sqrt() * x * f64::exp(-x.powi(2)))
            .collect();
        let tmp_D_ij: Vec<f64> = f_zeta_r_ij.iter().zip(r_scale.iter())
            .map(|(&f, &scale)| f * scale).collect();
        let mut D_ij = MatrixFull::from_vec([n_grids, n_grids], tmp_D_ij).unwrap();
        for i in 0..n_grids{
            let d_ii = -charge_exp[i]* (2.0 / PI).sqrt() / (2.0 * R_vdw[i]);
            D_ij[(i,i)] = d_ii;
        }

        (A, D_ij, S_ij)

    }

    pub fn get_K_R_f(surface: &SurfaceVdwGaussian, method: PcmMethod, epsilon: f64) -> (MatrixFull<f64>, MatrixFull<f64>, f64) {
        const PI: f64 = std::f64::consts::PI;

        //let device = A.device();
        let (A, D, S) = Self::get_A_D_S(&surface);
        let ngrids = A.len();
        let mut R = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
        for i in 0..ngrids {
            R[(i, i)] = 1.0 ;
        }
        //let A_matr = MatrixFull::from_vec([ngrids, 1], A.clone()).unwrap();

        match method {
            PcmMethod::CPCM => {
                let f_epsilon = (epsilon - 1.0) / epsilon;
                let K = S.clone();
                R.self_multiple(-f_epsilon);
                (K, R, f_epsilon)
            },
            PcmMethod::COSMO => {
                let f_epsilon = (epsilon - 1.0) / (epsilon + 0.5);
                let K = S.clone();
                R.self_multiple(-f_epsilon);
                (K, R, f_epsilon)
            },
            PcmMethod::IEFPCM => {
                let f_epsilon = (epsilon - 1.0) / (epsilon + 1.0);
                let mut DA = D.clone();
                for j in 0..ngrids{
                    for i in 0..ngrids{
                        DA[(i,j)] = DA[(i,j)] * A[j];
                    }
                }
                //let DA = _dgemm_scaled(&D, 'N', &A_matr, 'N', 1.0);
                let DAS = _dgemm_scaled(&DA, 'N', &S, 'N', 1.0);
                let K = S.scaled_add(&DAS, -f_epsilon / (2.0 * PI)).unwrap();
                //let K = S - f_epsilon / (2.0 * PI) * DAS;
                //let R = -f_epsilon * (rt::eye((ngrids, device)) - 1.0 / (2.0 * PI) * DA);
                R.self_scaled_add(&DA, - 1.0 / (2.0 * PI));
                R.self_multiple(-f_epsilon);
                (K, R, f_epsilon)
            },
            PcmMethod::SSVPE => {
                let f_epsilon = (epsilon - 1.0) / (epsilon + 1.0);
                let mut DA = D.clone();
                for j in 0..ngrids{
                    for i in 0..ngrids{
                        DA[(i,j)] = DA[(i,j)] * A[j];
                    }
                }
                //let DA =  _dgemm_scaled(&D, 'N', &A_matr, 'N', 1.0);;
                let mut DAS = _dgemm_scaled(&DA, 'N', &S, 'N', 1.0);
                DAS.self_add(&DAS.transpose());
                let K = S.scaled_add(&DAS,  - f_epsilon / (4.0 * PI)).unwrap();
                //let K = S - f_epsilon / (4.0 * PI) * (&DAS + DAS.t());
                R.self_scaled_add(&DA, - 1.0 / (2.0 * PI));
                R.self_multiple(-f_epsilon);
                //let R = -f_epsilon * (rt::eye((ngrids, device)) - 1.0 / (2.0 * PI) * DA);
                (K, R, f_epsilon)
            },
        }
    }

    pub fn get_v_grids_n(surface: &SurfaceVdwGaussian, cint_data: &CINTR2CDATA) -> Vec<f64> {
    //compute the nuclear contribution to the potential on the surface grid points
        let grid_coords = &surface.surface_calc.grid_coords;
        let charge_exp = &surface.surface_calc.charge_exp;
        let exponents = charge_exp.iter().map(|&x| x * x).collect::<Vec<f64>>();
        let atom_coords = &cint_data.atom_coords();
        let atom_charges = &cint_data.atom_charges();

        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords, exponents.as_slice());
        let fake_nuc_data = CINTR2CDATA::fakemol_for_charges(atom_coords, None);
        // Electron distribution satisfy the Gaussian distribution
        //g_i(\vec{r})=q_i(\xi_i^2/\pi)^{3/2}exp(-\xi_i^2|\vec{r}-\vec{s}_i|^2).
        // compute \iint g_i(r1) * 1/|r1 - r2| * dr1 dr2
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_nuc_data, &fake_chg_data], None, None).into();
        let tmpshape = [shape[0],shape[1]];
        let tmp_v_ng = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
        let mut v_ng = vec![];
        for j in 0..tmpshape[1]{
            let mut v_ng_j = 0.0;
            for i in 0..tmpshape[0]{
                v_ng_j  += tmp_v_ng[(i,j)] * atom_charges[i];
            }
            v_ng.push(v_ng_j);
        }
        //assert!(v_ng.len() == atom_charges.len(), "Length of v_ng{} must match number of grid points{}.", v_ng.len(), atom_charges.len());
        //let v_n = v_ng.iter()
        //    .zip(atom_charges.iter())
        //    .map(|(&v, &q)| v * q)
        //    .collect::<Vec<f64>>();
        v_ng
    }

    pub fn build_pcm_static(surface: &SurfaceVdwGaussian, cfg: &PcmObjectCfg, mol: &Molecule) -> PcmStatic {
        let mut cint_data = mol.initialize_cint(false);
        let (A, D, S) = Self::get_A_D_S(&surface);
        let (K, R, f_epsilon) = Self::get_K_R_f(&surface, cfg.method, cfg.epsilon);
        let v_grids_n = Self::get_v_grids_n(&surface, &cint_data);
        PcmStatic { A, D, S, K, R, f_epsilon, v_grids_n }
    }
}

/// Full PCM object containing configuration, surface data, and static PCM data
#[non_exhaustive]
#[derive(Clone)]
pub struct PcmObject {
    pub cfg: PcmObjectCfg,
    pub surface: SurfaceVdwGaussian,
    //pub cint_data: CINTR2CDATA,
    //pub intmd: HashMap<String, Tsr>,
    pub pstatic: PcmStatic,
    //pub pscf: PcmScf,
}

impl PcmObject {
    pub fn init_Pcm(cfg: PcmObjectCfg, surface: SurfaceVdwGaussian, pstatic: PcmStatic) -> Self {
        PcmObject { cfg, surface, pstatic }
    }

}

//=============================================================================
//  PCM SCF Structures
//=============================================================================

/// PCM data updated during SCF iterations
#[derive(Clone)]
pub struct PcmScf {
    pub v_grids_e: Vec<f64>,
    pub v_grids: Vec<f64>,
    pub veff: MatrixUpper<f64>,
    pub eng: f64,
    pub eng_nuc: f64,
    pub q_sym: MatrixFull<f64>,
}

impl PcmScf{
    pub fn get_v_grids_e(surface: &SurfaceVdwGaussian, cint_data: &CINTR2CDATA, dm: &Vec<MatrixFull<f64>>, spin_channel: &usize) -> Vec<f64> {
        
        let charge_exp = &surface.surface_calc.charge_exp;
        let grid_coords = &surface.surface_calc.grid_coords;
        
        let ngrids = charge_exp.len();
        //let mut v_grids_e: Tsr = rt::zeros(([ngrids], &device));
        let mut v_grids_e: Vec<f64> = vec![];

        const CHUNK: usize = 256;
        for p0 in (0..ngrids).step_by(CHUNK) {
            let p1 = (p0 + CHUNK).min(ngrids);
            let grid_coords_chunk = &grid_coords[p0..p1];
            let charge_exp_chunk = charge_exp[p0..p1].to_vec().iter().map(|x| x * x).collect::<Vec<f64>>();
            let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
            
            let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
            let tmpshape = [shape[0] * shape[1], shape[2]];
            let v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
            let mut tmp_v_e = vec![];
            for t in 0..shape[2]{
                let mut f_t = 0.0;
                for i_spin in 0..*spin_channel{
                    for j in 0..shape[1]{
                        for i in 0..shape[0]{
                            f_t  += v_nj[(j*shape[0]+i,t)] * dm[i_spin][(i,j)];
                        }
                    }
                }
                tmp_v_e.push(f_t);
            }

            v_grids_e.extend(tmp_v_e);
        }

        v_grids_e
    }

    pub fn get_veff_pcm_by_q(surface: &SurfaceVdwGaussian, cint_data: &CINTR2CDATA, q: &Vec<f64>) -> MatrixUpper<f64> {
        //let device = DeviceBLAS::default();
        let nao = cint_data.nao();
        let charge_exp = &surface.surface_calc.charge_exp;
        let grid_coords = &surface.surface_calc.grid_coords;
        //let grid_coords = grid_coords.chunks_exact(3).map(|chunk| [chunk[0], chunk[1], chunk[2]]).collect::<Vec<[f64; 3]>>();

        let ngrids = charge_exp.len();
        //let mut veff: Tsr = rt::zeros(([nao, nao].c(), &device));
        let mut veff = MatrixFull::<f64>::new([nao, nao], 0.0);

        const CHUNK: usize = 256;
        for p0 in (0..ngrids).step_by(CHUNK) {
            let p1 = (p0 + CHUNK).min(ngrids);
            let grid_coords_chunk = &grid_coords[p0..p1];
            let charge_exp_chunk = charge_exp[p0..p1].iter().map(|x| x * x).collect::<Vec<f64>>();
            //let charge_exp_chunk = charge_exp[p0..p1].to_vec();
            let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

            let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
            //println!("Block shape={:?}, expected=[{},{},{}]", shape, nao, nao, p1-p0);
            let tmpshape = [shape[0] * shape[1], shape[2]];
            let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
            let q_p =MatrixFull::from_vec([p1 - p0, 1], q[p0..p1].to_vec()).unwrap();
            let v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);  
            veff.iter_mut().zip(v_nj.iter()).for_each(|(ve, &vnj)| *ve -= vnj);

        }

        let veff_upper = veff.iter_matrixupper().unwrap().map(|&x| x).collect::<Vec<f64>>();
        let veff = MatrixUpper::from_vec(nao*(nao+1)/2 as usize, veff_upper).unwrap();
        veff
    }

    pub fn get_pcm_refresh(
        surface: &SurfaceVdwGaussian,
        mol: &Molecule,
        dm: &Vec<MatrixFull<f64>>,
        K: MatrixFull<f64>,
        R: MatrixFull<f64>,
        v_grids_n: Vec<f64>,
        spin_channel: &usize
    ) -> PcmScf {
        let mut cint_data = mol.initialize_cint(false);
        let v_grids_e = Self::get_v_grids_e(surface, &cint_data, dm, spin_channel);
        assert!(v_grids_e.len() == v_grids_n.len(), "Length mismatch: v_grids_e has {} elements but v_grids_n has {} elements", v_grids_e.len(), v_grids_n.len());
        let v_grids: Vec<f64> = v_grids_n.iter().zip(v_grids_e.iter()).map(|(&vn, &ve)| vn - ve).collect();
        let v_grids_matrix = MatrixFull::from_vec([v_grids.len(), 1], v_grids).unwrap();
        assert!(v_grids_matrix.size[0] == R.size[1], "Dimension mismatch: v_grids has {} rows but R has {} columns", v_grids_matrix.size[0], R.size[1]);
        let b = _dgemm_scaled(&R, 'N', &v_grids_matrix, 'N', 1.0);
        let q = solve_lu_no_inverse(&K, &b.data).unwrap();
        let q = MatrixFull::from_vec([q.len(), 1], q).unwrap();

        let vK_1 = solve_lu_no_inverse(&K.transpose(), &v_grids_matrix.data).unwrap();
        //let vK_1 = rt::linalg::solve_general((K.t(), v_grids.i((.., None)))).into_shape(-1);
        //let qt = R.t() % &vK_1;
        let vK_1 = MatrixFull::from_vec([vK_1.len(), 1], vK_1).unwrap();
        let qt = _dgemm_scaled(&R.transpose(), 'N', &vK_1, 'N', 1.0);
        if qt.size[0] != q.size[0] {
            panic!("Dimension mismatch: qt has {} rows but q has {} rows", qt.size[0], q.size[0]);
        }
        let q_sym = (q.clone() + qt.clone()) * 0.5;

        let veff = Self::get_veff_pcm_by_q(surface, &cint_data, &q_sym.data);
        let eng: f64 = 0.5 * q_sym.iter().zip(v_grids_matrix.iter())
            .map(|(&q, &v)| q * v)
            .sum::<f64>();
        //let eng = 0.5 * (&q_sym % &v_grids).to_scalar();
        let eng_nuc: f64 = 0.5 * q_sym.iter().zip(v_grids_n.iter())
            .map(|(&q, &vn)| q * vn)
            .sum::<f64>();
        let v_grids = v_grids_matrix.data;

        PcmScf{v_grids_e, v_grids, veff, eng, eng_nuc, q_sym}
    }

}   

/// Solve linear system Ax = b using LU decomposition without explicitly computing the inverse of A
pub fn solve_lu_no_inverse(a: &MatrixFull<f64>, b: &[f64]) -> Option<Vec<f64>>
{
    let n = a.size()[0];
    if n != b.len() || a.size()[0] != a.size()[1] { 
        return None; 
    }
    
    let mut lu_info = a.clone();
    let (lu, ipiv) = lu_info.to_matrixfullslicemut().lapack_dgetrf_full().unwrap();

    // 2. 应用行置换到右端项 b
    let mut x = b.to_vec();
    for i in 0..n {
        if ipiv[i] as usize != i + 1 {
            x.swap(i, ipiv[i] as usize - 1);
        }
    }
    
    // 3. 前向替换求解 L·y = Pb (L 是单位下三角)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lu[(i, j)] * x[j];
        }
        x[i] = (x[i] - sum); // L[i,i] = 1.0
    }
    
    // 4. 后向替换求解 U·x = y
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += lu[(i,j)] * x[j];
        }
        if lu[(i,i)].abs() < 1e-15 {
            println!("Singular matrix detected at diagonal element {}", i);
            return None;
        }
        x[i] = (x[i] - sum) / lu[(i,i)];
    }
    
    Some(x)
}

/// Main function to prepare PCM object for a given molecule
pub fn solvent_prepare(mol: &Molecule) -> PcmObject {
    let method = mol.solvent_model.clone();
    let epsilon = mol.epsilon.clone();
    let pcmcfg = PcmObjectCfg::build(method, epsilon);
    let surfacecfg = SurfaceVdwGaussianCfg::default();
    let mut surface = SurfaceVdwGaussian::new(surfacecfg, &mol.geom);
    surface.build();
    let pstatic = PcmStatic::build_pcm_static(&surface, &pcmcfg, &mol);
    //let pcm_object = PcmObject::init_Pcm(pcmcfg, surface, pstatic);
    PcmObject::init_Pcm(pcmcfg, surface, pstatic)
}


fn print_matrix_stats<T: std::fmt::Display + Copy>(matrix: &MatrixFull<T>) 
where
    T: std::fmt::Display + Copy + std::cmp::PartialOrd ,
{
    println!("size of matrix: {}x{}", matrix.size[0], matrix.size[1]);
    print_vec_stats(matrix.data.as_slice());
}

fn print_vec_stats<T: std::fmt::Display + Copy>(slice: &[T]) 
where 
    T: std::cmp::PartialOrd  // 使用 PartialOrd 而不是 Ord
{
    if slice.is_empty() {
        println!("empty vec");
        return;
    }
    
    // 使用 fold 处理浮点数
    let max = slice.iter()
        .fold(&slice[0], |a, b| if a > b { a } else { b });
    
    let min = slice.iter()
        .fold(&slice[0], |a, b| if a < b { a } else { b });
    
    println!("full length: {}", slice.len());
    println!("max: {}", max);
    println!("min: {}", min);
}

pub fn debug_print_pcm(sta: &PcmStatic, scf: &PcmScf){
    println!("A:");
    print_vec_stats(&sta.A);
    println!("vec of A:{:?}",sta.A);
    println!("D:");
    print_matrix_stats(&sta.D);
    println!("S:");
    print_matrix_stats(&sta.S);
    println!("K:");
    print_matrix_stats(&sta.K);
    println!("R:");
    print_matrix_stats(&sta.R);
    println!("v_grids_n:");
    print_vec_stats(&sta.v_grids_n);
    println!("v_grids_e:");
    print_vec_stats(&scf.v_grids_e);
    println!("q_sym:");
    print_matrix_stats(&scf.q_sym);
    println!("v_grids:");
    print_vec_stats(&scf.v_grids);
    println!("veff:");
    print_vec_stats(&scf.veff.data);
    println!("eng:");
    println!("{}", scf.eng);
}
//#[cfg(test)]
//mod tests {
//    #[test]
//    fn test_build_surface{
//
//    }
//}



/* #endregion */

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[test]
//    fn test_build_surface() {
//        let cint_data = cint_data_from_file();
//        let atom_ids = vec![8, 1, 1];
//        let atom_coords = cint_data.atom_coords();
//        let cfg_str = r#"
//            lebedev_degree = 302
//            vdw_scale = 1.2
//            switch_type = "SWIG"
//        "#;
//        let cfg: SurfaceVdwGaussianCfg = toml::from_str(cfg_str).unwrap();
//        let mut surface = SurfaceVdwGaussian::new(cfg, &atom_ids, &atom_coords);
//        surface.build();
//
//        assert!(tsr_mad(&surface.intmd["grid_coords"], &tensor_from_file("grid_coords.npy")) < 1e-8);
//        assert!(tsr_mad(&surface.intmd["norm_vec"], &tensor_from_file("norm_vec.npy")) < 1e-8);
//        assert!(tsr_mad(&surface.intmd["weights"], &tensor_from_file("weights.npy")) < 1e-8);
//        assert!(tsr_mad(&surface.intmd["charge_exp"], &tensor_from_file("charge_exp.npy")) < 1e-8);
//        assert!(tsr_mad(&surface.intmd["switch_fun"], &tensor_from_file("switch_fun.npy")) < 1e-8);
//        assert!(tsr_mad(&surface.intmd["area"], &tensor_from_file("area.npy")) < 1e-8);
//        assert!(tsr_mad(&surface.intmd["R_vdw"], &tensor_from_file("R_vdw.npy")) < 1e-8);
//
//        // check A, D, S
//        let (A, D, S) = get_A_D_S(&surface);
//        assert!(tsr_mad(&A, &tensor_from_file("A.npy")) < 1e-6);
//        assert!(tsr_mad(&D, &tensor_from_file("D.npy")) < 1e-6);
//        assert!(tsr_mad(&S, &tensor_from_file("S.npy")) < 1e-6);
//
//        // check K, R
//        let (K, R, _) = get_K_R_f(PcmMethod::CPCM, 78.3553, A.view(), D.view(), S.view());
//        assert!(tsr_mad(&K, &tensor_from_file("CPCM-K.npy")) < 1e-6);
//        assert!(tsr_mad(&R, &tensor_from_file("CPCM-R.npy")) < 1e-6);
//        let (K, R, _) = get_K_R_f(PcmMethod::COSMO, 78.3553, A.view(), D.view(), S.view());
//        assert!(tsr_mad(&K, &tensor_from_file("COSMO-K.npy")) < 1e-6);
//        assert!(tsr_mad(&R, &tensor_from_file("COSMO-R.npy")) < 1e-6);
//        let (K, R, _) = get_K_R_f(PcmMethod::IEFPCM, 78.3553, A.view(), D.view(), S.view());
//        assert!(tsr_mad(&K, &tensor_from_file("IEFPCM-K.npy")) < 1e-6);
//        assert!(tsr_mad(&R, &tensor_from_file("IEFPCM-R.npy")) < 1e-6);
//        let (K, R, _) = get_K_R_f(PcmMethod::SSVPE, 78.3553, A.view(), D.view(), S.view());
//        assert!(tsr_mad(&K, &tensor_from_file("SSVPE-K.npy")) < 1e-6);
//        assert!(tsr_mad(&R, &tensor_from_file("SSVPE-R.npy")) < 1e-6);
//
//        let (K, R, _) = get_K_R_f(PcmMethod::CPCM, 78.3553, A.view(), D.view(), S.view());
//        assert!(tsr_mad(&K, &tensor_from_file("K.npy")) < 1e-6);
//        assert!(tsr_mad(&R, &tensor_from_file("R.npy")) < 1e-6);
//
//        // check v_grids_n
//        let v_grids_n = get_v_grids_n(&surface, &cint_data);
//        assert!(tsr_mad(&v_grids_n, &tensor_from_file("v_grids_n.npy")) < 1e-6);
//
//        // check v_grids_e
//        let dm = tensor_from_file("dm.npy");
//        let v_grids_e = get_v_grids_e(&surface, &cint_data, dm.view());
//        assert!(tsr_mad(&v_grids_e, &tensor_from_file("v_grids_e.npy")) < 1e-6);
//
//        // check veff
//        let obj_intmd = get_veff_pcm(&surface, &cint_data, dm.view(), K.view(), R.view(), v_grids_e.view(), v_grids_n.view());
//        let epcm = obj_intmd["eng"].to_scalar();
//        println!("epcm: {}", epcm);
//        assert!((epcm - tensor_from_file("epcm.npy").to_scalar()).abs() < 1e-6);
//        assert!(tsr_mad(&obj_intmd["v_grids"], &tensor_from_file("v_grids.npy")) < 1e-6);
//        assert!(tsr_mad(&obj_intmd["q"], &tensor_from_file("q.npy")) < 1e-6);
//        assert!(tsr_mad(&obj_intmd["q_sym"], &tensor_from_file("q_sym.npy")) < 1e-6);
//        assert!(tsr_mad(&obj_intmd["veff"], &tensor_from_file("vmat.npy")) < 1e-6);
//    }
//
//    #[test]
//    fn test_build_surface_iswig() {
//        let cint_data = cint_data_from_file();
//        let atom_ids = vec![8, 1, 1];
//        let atom_coords = cint_data.atom_coords();
//        let cfg_surface: SurfaceVdwGaussianCfg = toml::from_str("").unwrap();
//        let cfg_pcm: PcmObjectCfg = toml::from_str("").unwrap();
//        let mut surface = SurfaceVdwGaussian::new(cfg_surface, &atom_ids, &atom_coords);
//        surface.build();
//
//        let mut pcm_object = PcmObject::new(cfg_pcm, surface, cint_data);
//
//        let dm = tensor_from_file("dm.npy");
//        let epcm = pcm_object.calc_eng(dm.view());
//        println!("epcm: {}", epcm);
//    }
//
//    fn cint_data_from_file() -> CINTR2CDATA {
//        let manifest_root = env!("CARGO_MANIFEST_DIR").to_string() + "/pyscf_ref/h2o-631gdp/";
//        let file_name = manifest_root + "mol.json";
//        CINTR2CDATA::from_json(&file_name)
//    }
//
//    fn tensor_from_file(fname: &str) -> Tsr {
//        // c-contiguous numpy array to f-contiguous rstsr
//        let manifest_root = env!("CARGO_MANIFEST_DIR").to_string() + "/pyscf_ref/h2o-631gdp/";
//        let bytes = std::fs::read(manifest_root + fname).unwrap();
//        let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
//        let shape = npy.shape().iter().map(|x| *x as usize).collect::<Vec<usize>>();
//        let data = npy.into_vec::<f64>().unwrap();
//        rt::asarray((data, shape, &DeviceBLAS::default()))
//    }
//
//    fn tsr_mad(a: &Tsr, b: &Tsr) -> f64 {
//        (a - b).abs().max()
//    }
//}
