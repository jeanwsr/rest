#![allow(non_snake_case)]
// Solvent gradient module
pub mod grad;
pub mod surface_utils;
pub mod smd_cds;
pub use surface_utils::*;

use std::fmt;
//use std::f64::consts;
use std::time::Instant;
//use rest_libcint::gto::prelude_dev::X;
//use rest_libcint::CintType;
use rest_libcint::prelude::*;
use rest_libcint_wrapper::*;
//use rstsr_core::prelude_dev::shape;
use rstsr::prelude::*;
use std::collections::HashMap;
use std::f64::consts;
use rest_libcint::gto::prelude_dev::X;
use rest_libcint::{CINTR2CDATA, CintType};
use serde::{Deserialize, Serialize};
use tensors::{map_full_to_upper, map_upper_to_full, MatrixFull, MatrixUpper, ri, BasicMatUp, BasicMatrix, MathMatrix, MatrixFullSlice, MatrixUpperSlice};
//use rstsr as rt;
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dgemm_scaled};
use crate::molecule_io::Molecule;
//use crate::geom_io::{GeomCell, get_mass_charge};
//use crate::constants::solvent as data;
//use crate::dft::gen_grids::angular_grid;
use rayon::prelude::*;
use rand::Rng;
//use rand::thread_rng;
use crate::utilities::memory_batch::*;

type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
fn shuffle<T>(slice: &mut [T], rng: &mut impl Rng) {
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0, i + 1);
        slice.swap(i, j);
    }
}
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
    /// SMD = IEFPCM (electrostatics) + CDS (cavitation-dispersion-solvent structure)
    SMD,
}

impl fmt::Display for PcmMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcmMethod::COSMO => write!(f, "COSMO"),
            PcmMethod::CPCM => write!(f, "CPCM"),
            PcmMethod::IEFPCM => write!(f, "IEFPCM"),
            PcmMethod::SSVPE => write!(f, "SSVPE"),
            PcmMethod::SMD => write!(f, "SMD"),
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
            "SMD" => Ok(PcmMethod::SMD),
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
    /// SMD solvent descriptors [n, n25, α, β, γ, ε, φ, ψ].
    /// Default: water. Only used when method == SMD.
    pub solvent_descriptors: [f64; 8],
    /// SMD solvent type: 1 = water (ICDS=1, pre-tabulated sigma), 2 = non-aqueous (ICDS=2).
    pub icds: i32,
}

impl PcmObjectCfg {
    pub fn build(method: PcmMethod, epsilon: f64, solvent_descriptors: [f64; 8], icds: i32) -> Self {
        PcmObjectCfg { method, epsilon, solvent_descriptors, icds }
    }
}

impl Default for PcmObjectCfg {
    fn default() -> Self {
        PcmObjectCfg {
            method: PcmMethod::default(),
            epsilon: 78.3553,
            solvent_descriptors: SMD_ERROR_DESCRIPTORS,
            icds: 0,
        }
    }
}

/// Water solvent descriptors for SMD: [n, n25, α, β, γ, ε, φ, ψ]
pub const SMD_WATER_DESCRIPTORS: [f64; 8] = [1.3328, 1.3323, 0.82, 0.35, -1.0, 78.355, -1.0, -1.0];

pub const SMD_ERROR_DESCRIPTORS: [f64; 8] = [-1.0; 8];

/// Fuzzy water check — only compares n, α, ε. Sentinel fields ignored.
fn is_water_descriptor(d: &[f64; 8]) -> bool {
    (d[0] - 1.3328).abs() < 0.01
    && (d[2] - 0.82).abs() < 0.01
    && (d[5] - 78.355).abs() < 1.0
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
//  PCM Static Structures
//=============================================================================

/// Static PCM data (computed once per calculation, geometry-dependent only)
#[derive(Clone)]
pub struct PcmStatic {
    pub A: Vec<f64>,
    pub D: MatrixFull<f64>,
    pub S: MatrixFull<f64>,
    pub K: MatrixFull<f64>,
    pub R: MatrixFull<f64>,
    pub f_epsilon: f64,
    pub v_grids_n: Vec<f64>,
    pub K_ipiv: Vec<i32>,
    pub K_initial: MatrixFull<f64>,
    /// SMD CDS energy (Hartree), total SASA (Å²), and gradient ([3] × [natm] Hartree/Bohr).
    /// Populated when method == SMD.
    pub e_cds: Option<f64>,
    pub tarea: Option<f64>,
    pub d_cds: Option<Vec<MatrixFull<f64>>>,
}

impl PcmStatic{
    pub fn build_pcm_static(surface: &SurfaceVdwGaussian, cfg: &PcmObjectCfg, mol: &Molecule) -> PcmStatic {
        let mut cint_data = mol.initialize_cint(false);
        let (A, D, S) = get_A_D_S(&surface);
        let (K, R, f_epsilon, K_ipiv, K_initial) = get_K_R_f(&surface, cfg.method, cfg.epsilon);
        let v_grids_n = get_v_grids_n(&surface, &cint_data);
        let (e_cds, tarea, d_cds) = if cfg.method == PcmMethod::SMD {
            let (e, a, d) = compute_cds_from_surface(surface, cfg);
            let natm = surface.atomic_num.len();
            let d_cds: Vec<MatrixFull<f64>> = (0..3).map(|dir| {
                let mut m = MatrixFull::new([natm, 1], 0.0);
                for i in 0..natm { m[[i, 0]] = d[i][dir]; }
                m
            }).collect();
            (Some(e), Some(a), Some(d_cds))
        } else {
            (None, None, None)
        };
        PcmStatic { A, D, S, K, R, f_epsilon, v_grids_n, K_ipiv, K_initial, e_cds, tarea, d_cds }
    }
}

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
            } else {
                r_ij.push(dist_3d(&grid_coords[i], &grid_coords[j]));
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
            n_r_ij.push(dx * norm_vec[j].0 + dy * norm_vec[j].1 + dz * norm_vec[j].2);
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

pub fn get_K_R_f(surface: &SurfaceVdwGaussian, method: PcmMethod, epsilon: f64) -> (MatrixFull<f64>, MatrixFull<f64>, f64, Vec<i32>, MatrixFull<f64>) {
    const PI: f64 = std::f64::consts::PI;

    //let device = A.device();
    let (A, D, S) = get_A_D_S(&surface);
    let ngrids = A.len();
    let mut R = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
    for i in 0..ngrids {
        R[(i, i)] = 1.0 ;
    }
    //let A_matr = MatrixFull::from_vec([ngrids, 1], A.clone()).unwrap();

    match method {
        PcmMethod::CPCM => {
            let f_epsilon = (epsilon - 1.0) / epsilon;
            let mut K_initial = S.clone();
            R.self_multiple(-f_epsilon);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
        },
        PcmMethod::COSMO => {
            let f_epsilon = (epsilon - 1.0) / (epsilon + 0.5);
            let mut K_initial = S.clone();
            R.self_multiple(-f_epsilon);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
        },
        PcmMethod::IEFPCM | PcmMethod::SMD => {
            let f_epsilon = (epsilon - 1.0) / (epsilon + 1.0);
            let mut DA = D.clone();
            for j in 0..ngrids{
                for i in 0..ngrids{
                    DA[(i,j)] = DA[(i,j)] * A[j];
                }
            }
            //let DA = _dgemm_scaled(&D, 'N', &A_matr, 'N', 1.0);
            let DAS = _dgemm_scaled(&DA, 'N', &S, 'N', 1.0);
            let mut K_initial = S.scaled_add(&DAS, -f_epsilon / (2.0 * PI)).unwrap();
            //let K = S - f_epsilon / (2.0 * PI) * DAS;
            //let R = -f_epsilon * (rt::eye((ngrids, device)) - 1.0 / (2.0 * PI) * DA);
            R.self_scaled_add(&DA, - 1.0 / (2.0 * PI));
            R.self_multiple(-f_epsilon);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
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
            let mut K_initial = S.scaled_add(&DAS,  - f_epsilon / (4.0 * PI)).unwrap();
            //let K = S - f_epsilon / (4.0 * PI) * (&DAS + DAS.t());
            R.self_scaled_add(&DA, - 1.0 / (2.0 * PI));
            R.self_multiple(-f_epsilon);
            //let R = -f_epsilon * (rt::eye((ngrids, device)) - 1.0 / (2.0 * PI) * DA);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
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

    v_ng
}


/// Full PCM object containing configuration, surface data, and static PCM data
#[non_exhaustive]
#[derive(Clone)]
pub struct PcmObject {
    pub cfg: PcmObjectCfg,
    pub surface: SurfaceVdwGaussian,
    pub pstatic: PcmStatic,
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
    pub fn get_pcm_refresh(
        surface: &SurfaceVdwGaussian,
        mol: &Molecule,
        dm: &Vec<MatrixFull<f64>>,
        K: &MatrixFull<f64>,
        K_ipiv: &Vec<i32>,
        R: &MatrixFull<f64>,
        v_grids_n: &Vec<f64>,
        spin_channel: &usize,
        max_memory: &Option<f64>,
        chunk_size: &usize,
        solv_ri: bool,
    ) -> PcmScf {
        let dt_solv0 = time::Local::now();
        // RI-based PCM: use auxiliary basis for (μν|g_i) integrals
        let mut cint_data_ri = mol.initialize_cint(true);
        let mut aux_cint = mol.make_auxmol_fake().initialize_cint(false);
        let mut cint_data = mol.initialize_cint(false);
        // Pre-build integral optimizer for 3c2e integrals
        //let opt_3c = cint_data_ri.optimizer("int3c2e");
        //cint_data_ri.c_opt = Some(Arc::new(opt_3c));

        //let opt_3c_1 = cint_data.optimizer("int3c2e");
        //cint_data.c_opt = Some(Arc::new(opt_3c_1));
        println!("nao= {}", cint_data_ri.nao());
        // Compute V_{PQ} = (P|Q) for auxiliary basis (computed once per SCF iteration)
        let nbas = mol.cint_bas.len() as i32;
        let nbas_aux = mol.cint_aux_bas.len() as i32;

        let (v_grids_e, v_grids_matrix, veff, q_sym, dt_4) = if solv_ri{

            let shls_slice_vpq = [[nbas, nbas + nbas_aux], [nbas, nbas + nbas_aux]];
            let (v_pq_out, v_pq_shape) = cint_data_ri.integral_s1::<int2c2e>(Some(&shls_slice_vpq));
            let mut v_pq = MatrixFull::from_vec([v_pq_shape[0], v_pq_shape[1]], v_pq_out).unwrap();
            let naux = v_pq_shape[0];
            println!("naux= {}", naux);
            // Regularize V_{PQ} to avoid numerical issues in Cholesky
            let v_diag_max = (0..naux).fold(0.0f64, |acc, i| acc.max(v_pq[(i, i)]));
            let lambda = 1e-12 * v_diag_max.max(1.0);
            for i in 0..naux {
                v_pq[(i, i)] += lambda;
            }
            println!("V_PQ regularization: lambda = {:10.6e}, diag_max = {:10.6e}", lambda, v_diag_max);
            let v_grids_e = get_v_grids_e(surface, mol, dm, spin_channel, max_memory, chunk_size,
                &cint_data_ri, &aux_cint, &v_pq);

            let dt_1 = time::Local::now();
            let timecost_vgrid_e = (dt_1.timestamp_millis()-dt_solv0.timestamp_millis()) as f64 /1000.0;
            println!("v_grids_e costs {:10.2} seconds.", timecost_vgrid_e);

            assert!(v_grids_e.len() == v_grids_n.len(), "Length mismatch: v_grids_e has {} elements but v_grids_n has {} elements", v_grids_e.len(), v_grids_n.len());
            let v_grids: Vec<f64> = v_grids_n.iter().zip(v_grids_e.iter()).map(|(&vn, &ve)| vn - ve).collect();
            let v_grids_matrix = MatrixFull::from_vec([v_grids.len(), 1], v_grids).unwrap();

            let dt_2 = time::Local::now();
            let timecost_vgrid = (dt_2.timestamp_millis()-dt_1.timestamp_millis()) as f64 /1000.0;
            println!("v_grids costs {:10.2} seconds.", timecost_vgrid);

            assert!(v_grids_matrix.size[0] == R.size[1], "Dimension mismatch: v_grids has {} rows but R has {} columns", v_grids_matrix.size[0], R.size[1]);
            let b = _dgemm_scaled(R, 'N', &v_grids_matrix, 'N', 1.0);
            let q = solve_lu(K, K_ipiv, &b.data).unwrap();
            let q = MatrixFull::from_vec([q.len(), 1], q).unwrap();

            //solve K^T x = v_grids
            let vK_1 = solve_lu_transpose(K, K_ipiv, &v_grids_matrix.data).unwrap();

            let vK_1 = MatrixFull::from_vec([vK_1.len(), 1], vK_1).unwrap();

            let dt_3 = time::Local::now();
            let timecost_solve = (dt_3.timestamp_millis()-dt_2.timestamp_millis()) as f64 /1000.0;
            println!("Solving linear equations costs {:10.2} seconds.", timecost_solve);

            let qt = _dgemm_scaled(&R.transpose(), 'N', &vK_1, 'N', 1.0);
            if qt.size[0] != q.size[0] {
            panic!("Dimension mismatch: qt has {} rows but q has {} rows", qt.size[0], q.size[0]);
            }
            let q_sym = (q + qt) * 0.5;

            let dt_4 = time::Local::now();
            let timecost_qsym = (dt_4.timestamp_millis()-dt_3.timestamp_millis()) as f64 /1000.0;
            println!("Symmetrization of q costs {:10.2} seconds.", timecost_qsym);

            let veff = get_veff_pcm_by_q(surface, mol, &q_sym.data, max_memory, chunk_size, &cint_data_ri, &aux_cint, &v_pq);
    
            (v_grids_e, v_grids_matrix, veff, q_sym, dt_4)
        } else {
            let v_grids_e = get_v_grids_e_old(surface, &cint_data, dm, spin_channel, max_memory, chunk_size);

            let dt_1 = time::Local::now();
            let timecost_vgrid_e = (dt_1.timestamp_millis()-dt_solv0.timestamp_millis()) as f64 /1000.0;
            println!("v_grids_e costs {:10.2} seconds.", timecost_vgrid_e);

            assert!(v_grids_e.len() == v_grids_n.len(), "Length mismatch: v_grids_e has {} elements but v_grids_n has {} elements", v_grids_e.len(), v_grids_n.len());
            let v_grids: Vec<f64> = v_grids_n.iter().zip(v_grids_e.iter()).map(|(&vn, &ve)| vn - ve).collect();
            let v_grids_matrix = MatrixFull::from_vec([v_grids.len(), 1], v_grids).unwrap();

            let dt_2 = time::Local::now();
            let timecost_vgrid = (dt_2.timestamp_millis()-dt_1.timestamp_millis()) as f64 /1000.0;
            println!("v_grids costs {:10.2} seconds.", timecost_vgrid);

            assert!(v_grids_matrix.size[0] == R.size[1], "Dimension mismatch: v_grids has {} rows but R has {} columns", v_grids_matrix.size[0], R.size[1]);
            let b = _dgemm_scaled(R, 'N', &v_grids_matrix, 'N', 1.0);
            let q = solve_lu(K, K_ipiv, &b.data).unwrap();
            let q = MatrixFull::from_vec([q.len(), 1], q).unwrap();

            //solve K^T x = v_grids
            let vK_1 = solve_lu_transpose(K, K_ipiv, &v_grids_matrix.data).unwrap();

            let vK_1 = MatrixFull::from_vec([vK_1.len(), 1], vK_1).unwrap();

            let dt_3 = time::Local::now();
            let timecost_solve = (dt_3.timestamp_millis()-dt_2.timestamp_millis()) as f64 /1000.0;
            println!("Solving linear equations costs {:10.2} seconds.", timecost_solve);

            let qt = _dgemm_scaled(&R.transpose(), 'N', &vK_1, 'N', 1.0);
            if qt.size[0] != q.size[0] {
            panic!("Dimension mismatch: qt has {} rows but q has {} rows", qt.size[0], q.size[0]);
            }
            let q_sym = (q + qt) * 0.5;

            let dt_4 = time::Local::now();
            let timecost_qsym = (dt_4.timestamp_millis()-dt_3.timestamp_millis()) as f64 /1000.0;
            println!("Symmetrization of q costs {:10.2} seconds.", timecost_qsym);

            let veff = get_veff_pcm_by_q_old(surface, &cint_data, &q_sym.data, max_memory, chunk_size);
            (v_grids_e, v_grids_matrix, veff, q_sym, dt_4)
        };

        let eng: f64 = 0.5 * q_sym.iter().zip(v_grids_matrix.iter())
            .map(|(&q, &v)| q * v)
            .sum::<f64>();

        let eng_nuc: f64 = 0.5 * q_sym.iter().zip(v_grids_n.iter())
            .map(|(&q, &vn)| q * vn)
            .sum::<f64>();
        let v_grids = v_grids_matrix.data;

        let dt_5 = time::Local::now();
        let timecost_veff = (dt_5.timestamp_millis()-dt_4.timestamp_millis()) as f64 /1000.0;
        println!("Computation of eng costs {:10.2} seconds.", timecost_veff);

        PcmScf{v_grids_e, v_grids, veff, eng, eng_nuc, q_sym}
    }

}   

pub fn get_v_grids_e_old(
    surface: &SurfaceVdwGaussian,
    cint_data: &CINTR2CDATA,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: &usize,
    max_memory: &Option<f64>,
    chunk_size: &usize) -> Vec<f64> {
        
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    
    let ngrids = charge_exp.len();


    //const CHUNK: usize = 16;
    let CHUNK =  *chunk_size;
    
    let nao = cint_data.nao();
    let mut dm_vec = vec![0.0; nao * nao];
    for i_spin in 0..*spin_channel {
        for j in 0..nao {
            for i in 0..nao {
                dm_vec[j*nao + i] += dm[i_spin][(i,j)];
            }
        }
    }
    let dm_mat = MatrixFull::from_vec([1, nao*nao], dm_vec).unwrap();
    
    /**
    let mem_avail_mb = max_memory.map(|max_memory| {
        max_memory - detect_used_memory_mb("proc")
    }).unwrap_or_else(detect_available_memory_mb);

    let CHUNK: usize = ((0.7 * 0.01 * mem_avail_mb * 1024.0 * 1024.0)
    / (8.0 * (nao * nao) as f64))
    .max(1.0) as usize;
    println!("Available memory for v_grids_e computation: {:.2} MB, chunk size: {}", mem_avail_mb, CHUNK);
    
    let mut v_grids_e = Vec::with_capacity(ngrids);
    for p0 in (0..ngrids).step_by(CHUNK) {
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let charge_exp_chunk = charge_exp[p0..p1].to_vec().iter().map(|x| x * x).collect::<Vec<f64>>();
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
        
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
        let tmpshape = [shape[0] * shape[1], shape[2]];
        let v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();

        let v_e_chunk = _dgemm_scaled(&dm_mat, 'N', &v_nj, 'N', 1.0);
        v_grids_e.extend(v_e_chunk.data);
    };
    **/

    let dt0 = time::Local::now();
    
    let mut v_grids_e = vec![0.0; ngrids];
    //let (sender, receiver) = channel();
    v_grids_e.par_chunks_mut(CHUNK).enumerate().for_each(|(v_chunk, idx)| {
        let t0 = Instant::now();
        let p0 = v_chunk * CHUNK;
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
        for &x in &charge_exp[p0..p1] {
            charge_exp_chunk.push(x * x);
        }
        //let charge_exp_chunk = charge_exp[p0..p1].to_vec().iter().map(|x| x * x).collect::<Vec<f64>>();
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
        
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
        let max_val = tmpout.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let v_e_chunk = if max_val < 1.0e-12 {
            MatrixFull::new([1, idx.len()], 0.0)
        } else {
            let tmpshape = [shape[0] * shape[1], shape[2]];
            let v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
            _dgemm_scaled(&dm_mat, 'N', &v_nj, 'N', 1.0)
        };  
        

        //println!("Chunk cycle of v_e costs {:10.6} seconds.", t0.elapsed().as_secs_f64());
        idx.iter_mut().zip(v_e_chunk.iter()).for_each(|(i, &v)| *i = v);

    });

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0;
    println!("Parallel computation of v_grids_e costs {:10.2} seconds.", timecost);
    
    
    v_grids_e
}

pub fn get_veff_pcm_by_q_old(
    surface: &SurfaceVdwGaussian,
    cint_data: &CINTR2CDATA,
    q: &Vec<f64>,
    max_memory: &Option<f64>,
    chunk_size: &usize) -> MatrixUpper<f64> {
    let nao = cint_data.nao();
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;

    let ngrids = charge_exp.len();

    //let mut veff = MatrixFull::<f64>::new([nao, nao], 0.0);
    
    let CHUNK = *chunk_size;
    /** 
    let mem_avail_mb = max_memory.map(|max_memory| {
        max_memory - detect_used_memory_mb("proc")
    }).unwrap_or_else(detect_available_memory_mb);

    let CHUNK: usize = ((0.7 * mem_avail_mb * 1024.0 * 1024.0)
    / (8.0 * (nao * nao) as f64))
    .max(1.0) as usize;
    println!("Available memory for veff computation: {:.2} MB, setting chunk size to {}", mem_avail_mb, CHUNK);


    for p0 in (0..ngrids).step_by(CHUNK) {
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let charge_exp_chunk = charge_exp[p0..p1].iter().map(|x| x * x).collect::<Vec<f64>>();
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
        //println!("Block shape={:?}, expected=[{},{},{}]", shape, nao, nao, p1-p0);
        let tmpshape = [shape[0] * shape[1], shape[2]];
        let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
        let q_p =MatrixFull::from_vec([p1 - p0, 1], q[p0..p1].to_vec()).unwrap();
        let v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);  
        veff.data.par_iter_mut().zip(v_nj.data.par_iter()).for_each(|(ve, &vnj)| *ve -= vnj);

    }
    */
    let dt0 = time::Local::now();
    /**
    let veff = (0..ngrids)
        .into_par_iter()
        .step_by(CHUNK)
        .map(|p0| {
            let t0 = Instant::now();
            let p1 = (p0 + CHUNK).min(ngrids);

            let grid_coords_chunk = &grid_coords[p0..p1];
            let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
            for &x in &charge_exp[p0..p1] {
                charge_exp_chunk.push(x * x);
            }
            //let charge_exp_chunk = charge_exp[p0..p1].iter().map(|x| x * x).collect::<Vec<f64>>();

            let fake_chg_data =
                CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

            let (tmpout, shape) =
                CINTR2CDATA::integrate_cross(
                    "int3c2e",
                    [cint_data, cint_data, &fake_chg_data],
                    None,
                    None,
                ).into();

            let tmpshape = [shape[0] * shape[1], shape[2]];
            let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();

            let q_p = MatrixFull::from_vec(
                [p1 - p0, 1],
                q[p0..p1].to_vec()
            ).unwrap();

            let mut v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);

            v_nj.data.iter_mut().for_each(|x| *x = -*x);

            println!("Chunk cycle of veff costs {:10.6} seconds.", t0.elapsed().as_secs_f64());
            v_nj
        })
        .reduce(
            || MatrixFull::<f64>::new([nao, nao], 0.0),
            |mut acc, v| {
                acc.data
                    .iter_mut()
                    .zip(v.data.iter())
                    .for_each(|(a, &b)| *a += b);
                acc
            }
        );
    */
    let veff = grid_coords
    .par_chunks(CHUNK)
    .zip(charge_exp.par_chunks(CHUNK))
    .zip(q.par_chunks(CHUNK))
    .map(|((grid_coords_chunk, charge_exp_chunk), q_chunk)| {
        let t0 = Instant::now();

        let chunk_len = grid_coords_chunk.len();

        let charge_exp_chunk_sq: Vec<f64> =
            charge_exp_chunk.iter().map(|x| x * x).collect();

        let fake_chg_data =
            CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk_sq.as_slice());

        let (tmpout, shape) =
            CINTR2CDATA::integrate_cross(
                "int3c2e",
                [cint_data, cint_data, &fake_chg_data],
                None,
                None,
            ).into();

        let tmpshape = [shape[0] * shape[1], shape[2]];
        
        let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();

        let q_p = MatrixFull::from_vec([chunk_len, 1],q_chunk.to_vec()).unwrap();

        let mut v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);

        v_nj.iter_mut().for_each(|x| *x = -*x);

        v_nj
    })
    .reduce(
        || MatrixFull::<f64>::new([nao, nao], 0.0),
        |mut acc, v| {
            acc.data
                .iter_mut()
                .zip(v.data.iter())
                .for_each(|(a, &b)| *a += b);
            acc
        }
    );

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis() - dt0.timestamp_millis()) as f64 / 1000.0;
    println!("Parallel computation of veff costs {:10.2} seconds.", timecost);
    let veff_upper = veff.iter_matrixupper().unwrap().map(|&x| x).collect::<Vec<f64>>();
    let veff = MatrixUpper::from_vec(nao*(nao+1)/2 as usize, veff_upper).unwrap();
    
    veff
}

//=============================================================================
//  RI-based PCM: get_v_grids_e (using auxiliary basis for (μν|g_i) integrals)
//=============================================================================

pub fn get_v_grids_e(
    surface: &SurfaceVdwGaussian,
    mol: &Molecule,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: &usize,
    max_memory: &Option<f64>,
    chunk_size: &usize,
    cint_data_ri: &CINTR2CDATA,
    aux_cint: &CINTR2CDATA,
    v_pq: &MatrixFull<f64>,
) -> Vec<f64> {
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let ngrids = charge_exp.len();
    let CHUNK = 256;
    let nao = dm[0].size[0] as usize;
    let naux = aux_cint.ao_loc().last().unwrap().clone();
    let nbas = mol.cint_bas.len() as i32;
    let nbas_aux = mol.cint_aux_bas.len() as i32;

    // Flatten DM to row vector [1, nao*nao]
    let mut dm_vec = vec![0.0; nao * nao];
    for i_spin in 0..*spin_channel {
        for j in 0..nao {
            for i in 0..nao {
                dm_vec[j*nao + i] += dm[i_spin][(i,j)];
            }
        }
    }
    let dm_mat = MatrixFull::from_vec([1, nao*nao], dm_vec).unwrap();

    // Partition auxiliary basis into batches
    let aux_loc = &cint_data_ri.ao_loc()[(nbas as usize)..];
    let aux_batch_size = 256;
    let partition = blocksize_partition(&aux_loc, aux_batch_size);

    let dt0 = time::Local::now();

    // Step 1: Compute C_P = Σ_{μν} D_{μν} · (μν|P) over aux batches
    let mut c_vec = vec![0.0; naux];
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        let nbatch = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
        let (out, shape) = cint_data_ri.integral_s1::<int3c2e>(Some(&shls_slice));
        // Debug: check integrals for NaN on first batch
        if idx_ao == 0 {
            let int_nan = out.iter().any(|&x| x.is_nan() || x.is_infinite());
            println!("RI-PCM: int3c shape={:?}, nbatch={}, int_nan={}, out[0..5]={:?}", shape, nbatch, int_nan, &out[..5.min(out.len())]);
        }
        let tmpshape = [nao * nao, nbatch];
        let int3c = MatrixFull::from_vec(tmpshape, out).unwrap();
        let c_batch = _dgemm_scaled(&dm_mat, 'N', &int3c, 'N', 1.0);
        for (i, &v) in c_batch.data.iter().enumerate() {
            c_vec[idx_ao + i] = v;
        }
        idx_ao += nbatch;
    }

    // Step 2: Solve V · Y = C via Cholesky decomposition (V is symmetric)
    let y_vec = solve_cholesky(&v_pq, &c_vec);
    let y_mat = MatrixFull::from_vec([naux, 1], y_vec).unwrap();

    // Debug: check step 1+2 results
    let c_minmax = c_vec.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let y_minmax = y_mat.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let c_nan = c_vec.iter().any(|&x| x.is_nan() || x.is_infinite());
    let y_nan = y_mat.data.iter().any(|&x| x.is_nan() || x.is_infinite());

    // Step 3: Parallel grid chunks — v_grids_e[j] = Σ_P Y_P · (P|g_j)
    let mut v_grids_e = vec![0.0; ngrids];
    v_grids_e.par_chunks_mut(CHUNK).enumerate().for_each(|(v_chunk, idx)| {
        let p0 = v_chunk * CHUNK;
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
        for &x in &charge_exp[p0..p1] {
            charge_exp_chunk.push(x * x);
        }
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
        
        //println!("CHUNK {}: exponents[0..5]={:?}, coef check: 2*α^1.5/π = {:?}",
        //v_chunk, &charge_exp_chunk[..5.min(charge_exp_chunk.len())],
        //charge_exp_chunk.iter().map(|&a| 2.0 * a.powf(1.5) / std::f64::consts::PI).take(5).collect::<Vec<_>>());

        //actual code let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, aux_cint], None, None).into();
        //let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, cint_data_ri], None, None).into();
        //let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, &fake_chg_data], None, None).into();
        let shls_slice_2c = [[0, (p1-p0) as i32], [nbas, nbas + nbas_aux]];
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, cint_data_ri], None, &shls_slice_2c).into();
        let out_nan = tmpout.iter().any(|&x| x.is_nan() || x.is_infinite());
        let out_minmax = tmpout.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        println!("CHUNK {}: RAW int2c2e shape={:?}, bad={}, min={:10.6e}, max={:10.6e}, first5={:?}",
                 v_chunk, shape, out_nan, out_minmax.0, out_minmax.1, &tmpout[..5.min(tmpout.len())]);
        let pg_shape = [shape[0], shape[1]];
        let pg_mat = MatrixFull::from_vec(pg_shape, tmpout).unwrap();
        let pg_nan = pg_mat.data.iter().any(|&x| x.is_nan() || x.is_infinite());
        let pg_minmax = pg_mat.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let v_e_chunk = _dgemm_scaled(&pg_mat, 'N', &y_mat, 'N', 1.0);
        idx.iter_mut().zip(v_e_chunk.iter()).for_each(|(i, &v)| *i = v);
    });

    // Debug: check v_grids_e results
    let ve_minmax = v_grids_e.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let ve_nan = v_grids_e.iter().any(|&x| x.is_nan() || x.is_infinite());
    println!("RI-PCM: v_grids_e: min={:10.6e}, max={:10.6e}, bad={}", ve_minmax.0, ve_minmax.1, ve_nan);

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis() - dt0.timestamp_millis()) as f64 / 1000.0;
    println!("RI-PCM: v_grids_e costs {:10.2} seconds.", timecost);

    v_grids_e
}

//=============================================================================
//  RI-based PCM: get_veff_pcm_by_q (using auxiliary basis for (μν|g_i) integrals)
//=============================================================================

pub fn get_veff_pcm_by_q(
    surface: &SurfaceVdwGaussian,
    mol: &Molecule,
    q: &Vec<f64>,
    max_memory: &Option<f64>,
    chunk_size: &usize,
    cint_data_ri: &CINTR2CDATA,
    aux_cint: &CINTR2CDATA,
    v_pq: &MatrixFull<f64>,
) -> MatrixUpper<f64> {

    let nbas = mol.cint_bas.len() as i32;
    let nbas_aux = mol.cint_aux_bas.len() as i32;
    let nao = cint_data_ri.ao_loc()[nbas as usize].clone(); 
    let naux = aux_cint.ao_loc().last().unwrap().clone();
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let ngrids = charge_exp.len();
    let CHUNK = *chunk_size;

    // Auxiliary basis info
    let aux_loc = &cint_data_ri.ao_loc()[(nbas as usize)..];

    let dt0 = time::Local::now();

    // Step 1: Compute Z'_P = Σ_j q_j · (P|g_j) over grid chunks (parallel)
    let z_prime = grid_coords
        .par_chunks(CHUNK)
        .zip(charge_exp.par_chunks(CHUNK))
        .zip(q.par_chunks(CHUNK))
        .map(|((grid_coords_chunk, charge_exp_chunk), q_chunk)| {
            let chunk_len = grid_coords_chunk.len();
            let charge_exp_chunk_sq: Vec<f64> =
                charge_exp_chunk.iter().map(|x| x * x).collect();
            let fake_chg_data =
                CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk_sq.as_slice());

            //let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [aux_cint, &fake_chg_data], None, None).into();
            let shls_slice_2c = [[nbas, nbas + nbas_aux], [0, chunk_len as i32]];
            let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [cint_data_ri, &fake_chg_data], None, &shls_slice_2c).into();
            let pg_shape = [shape[0], shape[1]];
            let pg_mat = MatrixFull::from_vec(pg_shape, tmpout).unwrap();
            let q_mat = MatrixFull::from_vec([chunk_len, 1], q_chunk.to_vec()).unwrap();
            let z_contrib = _dgemm_scaled(&pg_mat, 'N', &q_mat, 'N', 1.0);
            z_contrib.data
        })
        .reduce(
            || vec![0.0; naux],
            |mut a, b| {
                for (i, &v) in b.iter().enumerate() {
                    a[i] += v;
                }
                a
            });

    // Step 2: Solve V · Z = Z' via Cholesky decomposition (V is symmetric)
    let z_vec = solve_cholesky(&v_pq, &z_prime);

    // Step 3: Build veff[μ,ν] += -(μν|P) · Z_P over aux batches
    let aux_batch_size = 256;
    let partition = blocksize_partition(&aux_loc, aux_batch_size);
    let mut veff = MatrixFull::<f64>::new([nao, nao], 0.0);
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        let nbatch = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
        let (out, shape) = cint_data_ri.integral_s1::<int3c2e>(Some(&shls_slice));
        let tmpshape = [nao * nao, nbatch];
        let int3c = MatrixFull::from_vec(tmpshape, out).unwrap();
        let z_batch = MatrixFull::from_vec([nbatch, 1], z_vec[idx_ao..idx_ao + nbatch].to_vec()).unwrap();
        let contrib = _dgemm_scaled(&int3c, 'N', &z_batch, 'N', 1.0);
        for (i, &v) in contrib.data.iter().enumerate() {
            veff.data[i] -= v;
        }
        idx_ao += nbatch;
    }

    // Pack to MatrixUpper
    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis() - dt0.timestamp_millis()) as f64 / 1000.0;
    println!("RI-PCM: veff costs {:10.2} seconds.", timecost);

    // Debug: check veff for NaN
    let veff_nan = veff.data.iter().any(|&x| x.is_nan() || x.is_infinite());
    if veff_nan {
        println!("WARNING: veff contains NaN or Inf!");
    } else {
        let vf_minmax = veff.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        println!("RI-PCM: veff data min={:10.6e}, max={:10.6e}", vf_minmax.0, vf_minmax.1);
    }

    let veff_upper = veff.iter_matrixupper().unwrap().map(|&x| x).collect::<Vec<f64>>();
    MatrixUpper::from_vec(nao * (nao + 1) / 2, veff_upper).unwrap()
}


/// Main function to prepare PCM object for a given molecule
pub fn solvent_prepare(mol: &Molecule) -> PcmObject {
    let method = mol.ctrl.solvent_model.clone();
    let epsilon = mol.ctrl.solv_epsilon;
    
    let descriptors = {
        if method == PcmMethod::SMD {
            mol.ctrl.solvent_descriptors.unwrap_or_else(|| {
                panic!("SMD solvent model requires solvent descriptors. You should set solvent_descriptors or solvent_name in the control file.");
            })
        } 
        else {SMD_ERROR_DESCRIPTORS}
    };
    let icds = if is_water_descriptor(&descriptors) { 1 } else { 2 };

    if method == PcmMethod::SMD && icds == 2 {
        if descriptors[0] < 0.0 || descriptors[2] < 0.0 || descriptors[3] < 0.0
            || descriptors[4] < 0.0 || descriptors[6] < 0.0 || descriptors[7] < 0.0
        {
            panic!("SMD solvent model requires valid [n, α, β, γ, φ, ψ]. \
                   Found negative value in them. \
                   Check solvent_descriptors or solvent_name in the control file.");
        }
    }
    let pcmcfg = PcmObjectCfg::build(method, epsilon, descriptors, icds);

    // Build cavity surface
    let mut surface = SurfaceVdwGaussian::new(mol.ctrl.pcm_cavity_radii, &mol.geom);
    if method == PcmMethod::SMD {
        // SMD uses intrinsic atomic Coulomb radii (eq. 16, Marenich 2009),
        // no vdW scaling (scale = 1.0)
        let alpha = descriptors[2]; // H-bond acidity
        surface.cfg.atom_radii = Some(smd_radii(alpha, &surface.atomic_num));
        surface.cfg.vdw_scale = Some(1.0);
    }
    surface.build();

    let pstatic = PcmStatic::build_pcm_static(&surface, &pcmcfg, &mol);
    PcmObject::init_Pcm(pcmcfg, surface, pstatic)
}

/// Compute SMD CDS energy and gradient from surface data.
/// Returns (gcds_hartree, tarea_ang2, dcds_hartree_per_bohr).
pub fn compute_cds_from_surface(
    surface: &SurfaceVdwGaussian,
    cfg: &PcmObjectCfg,
) -> (f64, f64, Vec<[f64; 3]>) {
    let natm = surface.atomic_num.len();
    let atomic_numbers = surface.atomic_num.clone();
    // Convert [3, natm] MatrixFull → [natm, 3] Vec<[f64;3]>
    let mut coords = vec![[0.0f64; 3]; natm];
    for i in 0..natm {
        coords[i] = [
            surface.atom_coords[[0, i]],
            surface.atom_coords[[1, i]],
            surface.atom_coords[[2, i]],
        ];
    }
    smd_cds::compute_cds(&atomic_numbers, &coords, cfg.icds, &cfg.solvent_descriptors)
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
    //println!("vec of A:{:?}",sta.A);
    println!("D:");
    print_matrix_stats(&sta.D);
    println!("S:");
    print_matrix_stats(&sta.S);
    println!("K:");
    print_matrix_stats(&sta.K_initial);
    println!("R:");
    print_matrix_stats(&sta.R);
    println!("v_grids_n:");
    print_vec_stats(&sta.v_grids_n);
    println!("v_grids_e:");
    print_vec_stats(&scf.v_grids_e);
    println!("q_sym:");
    print_matrix_stats(&scf.q_sym);
    println!("abs(q_sym):");
    print_vec_stats(&scf.q_sym.data.iter().map(|x| x.abs()).collect::<Vec<f64>>());
    println!("v_grids:");
    print_vec_stats(&scf.v_grids);
    println!("veff:");
    print_vec_stats(&scf.veff.data);
    println!("eng:");
    println!("{}", scf.eng);
}
