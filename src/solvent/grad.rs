//! Analytical gradients for PCM solvent model.
//!
//! Three components:
//! - **grad_nuc**:  derivative of the nuclear potential  (int2c2e_ip1)
//! - **grad_qv**:   derivative of the electronic potential (int3c2e_ip1/ip2)
//! - **grad_solver**: cavity response (dF/dA → dS/dD → branch per PCM method)
//!
//! ```text
//! dE_PCM/dx = q_sym · d(v_nuc − v_e)/dx + ½ v^T d(K⁻¹R)/dx v
//!           = grad_nuc + grad_qv + grad_solver
//! ```
//! Reference: J. Chem. Phys. 133, 244111 (2010), Appendix C.

#![allow(non_snake_case)]

use std::f64::consts::PI;
use std::time::Instant;

use rest_libcint::CINTR2CDATA;
use tensors::MatrixFull;
use tensors::matrix_blas_lapack::_dgemm_scaled;
use crate::geom_io::get_mass_charge;
use crate::Molecule;
use crate::scf_io::SCF;
use crate::constants::solvent as data;
use crate::solvent::{
    self,
    PcmMethod, PcmStatic, PcmScf, SurfaceVdwGaussian,
    solve_lu_transpose,
    RadiusScheme,
};

// ============================================================================
// Switch function: h(x) first derivative
// ============================================================================

fn ief_debug_enabled() -> bool {
    std::env::var("REST_IEF_DEBUG").map_or(false, |v| v == "1")
}

fn ief_print_grad(label: &str, de: &MatrixFull<f64>) {
    println!("IEF_DBG| {} [3,{}]", label, de.size[1]);
    for a in 0..de.size[1] {
        println!("IEF_DBG|   [{a}] = {:.12e} {:.12e} {:.12e}",
            de[(0, a)], de[(1, a)], de[(2, a)]);
    }
}

/// First derivative dh/dx of the PCM switch‑function.
///
/// h(x) = 10x³ − 15x⁴ + 6x⁵  (0 ≤ x ≤ 1)
/// dh/dx = 30x² − 60x³ + 30x⁴
pub fn grad_switch_h(x: f64) -> f64 {
    if x < 0.0 || x >= 1.0 { 0.0 }
    else { 30.0 * x.powi(2) - 60.0 * x.powi(3) + 30.0 * x.powi(4) }
}

// ============================================================================
// get_dF_dA   Appendix C  of J. Chem. Phys. 133, 244111 (2010)
// ============================================================================

/// Derivative of the cavity‑surface switch‑function F and area A
/// with respect to nuclear coordinates.
///
/// # Returns
/// - `dF` [`ngrids, natm×3`]: ∂F_g / ∂R_ia[xyz]
/// - `dA` [`ngrids, natm×3`]: ∂A_g / ∂R_ia[xyz]
///
/// For each grid point g contributed by atom A:
/// ```text
///   F_g = Σ_J h(d_gJ)   (switch function sum over overlapping atoms J)
///   dF_g / dR_A = Σ_J dh/dR_J · δ_AJ   (grid response + atom response)
/// ```
/// Reference: J. Chem. Phys. 133, 244111 (2010), Appendix C.
pub fn get_dF_dA(
    surface: &SurfaceVdwGaussian,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let natm = surface.atomic_num.len();
    let atom_coords = &surface.atom_coords;              // [3, natm]
    let grid_coords = &surface.surface_calc.grid_coords;  // Vec<[f64; 3]>
    let switch_fun = &surface.surface_calc.switch_fun;
    let area = &surface.surface_calc.area;
    let ngrids = grid_coords.len();
    let gslice = &surface.gslice_by_atom;
    let lebedev = surface.cfg.lebedev_degree;

    let vdw_scale = surface.cfg.effective_vdw_scale();
        let base_radii: &[f64] = match surface.cfg.radius_scheme {
            RadiusScheme::Bondi => &data::VDW_RADII,
            RadiusScheme::UFF   => &data::UFF_RADII,
        };
    // Per‑atom radii;  R_in / R_sw follow the same logic as build()
    let atom_radii = surface.cfg.atom_radii.clone().unwrap_or_else(|| {
        surface.atomic_num.iter()
            .map(|&z| {
                let mut r0 = base_radii[z];
                // Bondi 对 H 有特殊的半径值 (1.1 Å vs 表里的 1.2 Å)
                if z == 1 && surface.cfg.radius_scheme == RadiusScheme::Bondi {
                    r0 = 2.0786987370215684;
                }
                r0 * vdw_scale
            })
            .collect()
    });

    let R_sw: Vec<f64> = atom_radii.iter()
        .map(|&r| r * (14.0 / (lebedev as f64)).sqrt())
        .collect();
    let R_in: Vec<f64> = atom_radii.iter().zip(R_sw.iter())
        .map(|(&rv, &rs)| {
            let a = 0.5 + rv / rs - ((rv / rs).powi(2) - 1.0 / 28.0).sqrt();
            rv - a * rs
        })
        .collect();

    // dF[(g, ia*3 + xyz)] = ∂F_g / ∂R_ia[xyz]
    let mut dF = MatrixFull::<f64>::new([ngrids, natm * 3], 0.0);
    // dA[(g, ia*3 + xyz)] = ∂A_g / ∂R_ia[xyz]
    let mut dA = MatrixFull::<f64>::new([ngrids, natm * 3], 0.0);

    //calculate in the order of grids belonging to each atom 
    // so grid response exists only when dr_iJ/dR_ia, where i'th grid belonging to ia, and atom response exists only when dr_iJ/dR_J, and it exists for all i 
    for ia in 0..natm {
        let (p0, p1) = gslice[ia];
        let n = p1 - p0;
        if n == 0 { continue; }

        // --- distance vectors and switch‑function derivatives ---
        // r_{iJ} = |grid_i – coord_J|
        let mut riJ = MatrixFull::<f64>::new([n, natm], 0.0);
        // (dx, dy, dz) per grid–atom pair, stored as three [n, natm] matrices
        let mut ri_vec = [MatrixFull::<f64>::new([n, natm], 0.0),
                          MatrixFull::<f64>::new([n, natm], 0.0),
                          MatrixFull::<f64>::new([n, natm], 0.0)];
        // d_{iJ} = (r_{iJ} - R_in[J]) / R_sw[J]   (reduced distance)
        let mut diJ = MatrixFull::<f64>::new([n, natm], 0.0);

        for (ig, g) in (p0..p1).enumerate() {
            let gx = grid_coords[g][0];
            let gy = grid_coords[g][1];
            let gz = grid_coords[g][2];
            for j in 0..natm {
                let dx = gx - atom_coords[(0, j)];
                let dy = gy - atom_coords[(1, j)];
                let dz = gz - atom_coords[(2, j)];
                let r  = (dx*dx + dy*dy + dz*dz).sqrt();
                riJ[(ig, j)] = r;
                ri_vec[0][(ig, j)] = dx;
                ri_vec[1][(ig, j)] = dy;
                ri_vec[2][(ig, j)] = dz;
                diJ[(ig, j)] = if j == ia { 1.0 }
                               else if r < 1e-8 { 0.0 }
                               else { (r - R_in[j]) / R_sw[j] };
            }
        }

        // dh/dR_J[xyz] = h'(d) / (h(d) · r · R_sw) · (dx, dy, dz)
        // weighted gradient of the switch function
        let mut df_w = [MatrixFull::<f64>::new([n, natm], 0.0),
                        MatrixFull::<f64>::new([n, natm], 0.0),
                        MatrixFull::<f64>::new([n, natm], 0.0)];
        for ig in 0..n {
            for j in 0..natm {
                let d = diJ[(ig, j)];
                if d < 1e-8 { continue; }
                let fj = solvent::switch_h(d);
                let df = grad_switch_h(d);
                let r  = riJ[(ig, j)];
                if fj.abs() <= 1e-15 || r <= 1e-15 { continue; }
                // dh(v)/dR_J = h'(d) / (h(d) · r · R_sw) · (r - R_J) / r
                let base = df / (fj * r * R_sw[j]);
                df_w[0][(ig, j)] = base * ri_vec[0][(ig, j)];
                df_w[1][(ig, j)] = base * ri_vec[1][(ig, j)];
                df_w[2][(ig, j)] = base * ri_vec[2][(ig, j)];
            }
        }

        // ------ grid response: dF/dR_A for grid points belonging to A ------
        // (s_x, s_y, s_z) = Σ_J dh(v_gJ)/dR_J  (sum over all atoms)
        //For grids belonging to the ia'th atom, sum dr_iJ/dR_ia over all J 
        // dr_iJ/dR_ia = (s_i - R_j)/r_iJ, where s_i is the coord of the i'th grid point which belongs to ia'th atom
        for ig in 0..n {
            let gi = p0 + ig;
            let Fi = switch_fun[gi];
            let Ai = area[gi];
            let mut sx = 0.0; let mut sy = 0.0; let mut sz = 0.0;
            for j in 0..natm {
                sx += df_w[0][(ig, j)];
                sy += df_w[1][(ig, j)];
                sz += df_w[2][(ig, j)];
            }
            dF[(gi, ia*3 + 0)] += Fi * sx;
            dF[(gi, ia*3 + 1)] += Fi * sy;
            dF[(gi, ia*3 + 2)] += Fi * sz;
            dA[(gi, ia*3 + 0)] += Ai * sx;
            dA[(gi, ia*3 + 1)] += Ai * sy;
            dA[(gi, ia*3 + 2)] += Ai * sz;
        }

        // ------ atom response: dF/dR_J for grid points of A ------
        // (atom J receives opposite sign from grid response for trans. inv.)
        // dr_iJ/dR_A, when A = J, has an additional term: -(s_i - R_J)/r_iJ
        for ig in 0..n {
            let gi = p0 + ig;
            let Fi = switch_fun[gi];
            let Ai = area[gi];
            for j in 0..natm {
                dF[(gi, j*3 + 0)] -= Fi * df_w[0][(ig, j)];
                dF[(gi, j*3 + 1)] -= Fi * df_w[1][(ig, j)];
                dF[(gi, j*3 + 2)] -= Fi * df_w[2][(ig, j)];
                dA[(gi, j*3 + 0)] -= Ai * df_w[0][(ig, j)];
                dA[(gi, j*3 + 1)] -= Ai * df_w[1][(ig, j)];
                dA[(gi, j*3 + 2)] -= Ai * df_w[2][(ig, j)];
            }
        }
    }

    if ief_debug_enabled() {
        // per-atom RMS of dF and dA
        eprintln!("IEF_DBG| dF-dA per-atom rms (3*natm):");
        for a in 0..natm {
            let mut dF_rms = [0.0f64; 3]; let mut dA_rms = [0.0f64; 3];
            for xyz in 0..3 {
                let mut s_f = 0.0; let mut s_a = 0.0;
                for i in 0..dF.size[0] {
                    s_f += dF[(i, a*3+xyz)].powi(2);
                    s_a += dA[(i, a*3+xyz)].powi(2);
                }
                dF_rms[xyz] = (s_f / dF.size[0] as f64).sqrt();
                dA_rms[xyz] = (s_a / dA.size[0] as f64).sqrt();
            }
            eprintln!("IEF_DBG|   atom[{a}] dF_rms=({:.6e},{:.6e},{:.6e}) dA_rms=({:.6e},{:.6e},{:.6e})",
                dF_rms[0], dF_rms[1], dF_rms[2], dA_rms[0], dA_rms[1], dA_rms[2]);
        }
    }

    (dF, dA)
}

// ============================================================================
// get_dD_dS
// ============================================================================

/// Derivative of the S (and optionally D) matrix w.r.t. grid coordinates.
///
/// # Returns
/// - `dD` [`Some(Vec<MatrixFull<f64>>)`] — D derivatives per direction [3, ngrids, ngrids]
/// - `dS` [`Vec<MatrixFull<f64>>`] — S derivatives per direction [3, ngrids, ngrids]
/// - `dSii_dF` [`Vec<f64>`] — per‑grid coefficient for diagonal correction [ngrids],
///   ∂S_ii/∂R_a[xyz] = dSii_dF[i] · dF[(i, a*3+xyz)]
///
/// The S-matrix derivative satisfies:
/// ```text
///   ∂S_ij/∂r_i[xyz] = fac · (r_i - r_j)_xyz / r
///   ∂S_ji/∂r_i[xyz] = -fac · (r_i - r_j)_xyz / r   (translational invariance)
/// ```
pub fn get_dD_dS(
    surface: &SurfaceVdwGaussian,
    dF: &MatrixFull<f64>,
    with_D: bool,
) -> (Option<Vec<MatrixFull<f64>>>,
      Vec<MatrixFull<f64>>,
      Vec<f64>) {
    let grid_coords = &surface.surface_calc.grid_coords;
    let exponents   = &surface.surface_calc.charge_exp;
    let norm_vec    = &surface.surface_calc.norm_vec;
    let switch_fun  = &surface.surface_calc.switch_fun;
    let ngrids = grid_coords.len();

    // ---- dS: ∂S_ij/∂r_i[xyz] ----
    let mut dS: Vec<MatrixFull<f64>> = (0..3)
        .map(|_| MatrixFull::<f64>::new([ngrids, ngrids], 0.0))
        .collect();

    for i in 0..ngrids {
        let xi = exponents[i];
        let (gix, giy, giz) = (grid_coords[i][0], grid_coords[i][1], grid_coords[i][2]);
        for j in 0..ngrids {
            if i == j { continue; }
            let (gjx, gjy, gjz) = (grid_coords[j][0], grid_coords[j][1], grid_coords[j][2]);
            let dx = gix - gjx; let dy = giy - gjy; let dz = giz - gjz;
            let r2 = dx*dx + dy*dy + dz*dz;
            let r  = r2.sqrt();
            if r < 1e-15 { continue; }

            // x_{ij} = ξ_i·ξ_j / √(ξ_i² + ξ_j²)
            let xij = xi * exponents[j] / (xi*xi + exponents[j]*exponents[j]).sqrt();
            let xr  = xij * r;
            let erf = libm::erf(xr);
            let expv = f64::exp(-xr*xr);
            // dS/dr = -(erf(xr) - 2xr/√π · exp(-xr²)) / r²
            let dSdr = -(erf - 2.0 * xr / PI.sqrt() * expv) / r2;
            let fac = dSdr / r;
            for xyz in 0..3 {
                let d = match xyz { 0 => dx, 1 => dy, _ => dz };
                dS[xyz][(i, j)] = fac * d;
                dS[xyz][(j, i)] = -fac * d;   // translation invariance
            }
        }
    }

    // ---- dSii_dF: per-grid-point coefficient for diagonal correction ----
    // ∂S_ii/∂R_a[xyz] = dSii_dF[i] · dF[(i, a*3+xyz)]
    // where dSii_dF[i] = -ξ_i · √(2/π) / F_i²
    let sqrt2pi = (2.0 / PI).sqrt();
    let dSii_dF: Vec<f64> = (0..ngrids)
        .map(|i| -exponents[i] * sqrt2pi / (switch_fun[i] * switch_fun[i]))
        .collect();

    // ---- dD (only for IEF-PCM / SS(V)PE) ----
    let dD = if with_D {
        let mut dD_mat: Vec<MatrixFull<f64>> = (0..3)
            .map(|_| MatrixFull::<f64>::new([ngrids, ngrids], 0.0))
            .collect();
        for i in 0..ngrids {
            let xi = exponents[i];
            let (gix, giy, giz) = (grid_coords[i][0], grid_coords[i][1], grid_coords[i][2]);
            for j in 0..ngrids {
                if i == j { continue; }
                // n_j: normal vector at grid point j (D_ij is the normal derivative at j)
                let (njx, njy, njz) = (norm_vec[j].0, norm_vec[j].1, norm_vec[j].2);
                let (gjx, gjy, gjz) = (grid_coords[j][0], grid_coords[j][1], grid_coords[j][2]);
                let dx = gix - gjx; let dy = giy - gjy; let dz = giz - gjz;
                let r2 = dx*dx + dy*dy + dz*dz;
                let r  = r2.sqrt();
                if r < 1e-15 { continue; }
                let xij = xi * exponents[j] / (xi*xi + exponents[j]*exponents[j]).sqrt();
                let xr  = xij * r;
                let expv = f64::exp(-xr*xr);
                let erf = libm::erf(xr);
                let dSdr0 = -(erf - 2.0*xr/PI.sqrt()*expv) / r2;
                // n_j · (r_i - r_j)
                let nj_rij = njx*dx + njy*dy + njz*dz;
                let dD_dri = 4.0 * xr*xr * xij / PI.sqrt() * expv
                           * nj_rij / (r2 * r);
                for xyz in 0..3 {
                    let d = match xyz { 0 => dx, 1 => dy, _ => dz };
                    let dr = d / r;
                    let nc = match xyz { 0 => njx, 1 => njy, _ => njz };
                    dD_mat[xyz][(i, j)] = dD_dri * dr
                        + dSdr0 * (-nc/r + 3.0*nj_rij/r2 * dr);
                }
            }
        }
        Some(dD_mat)
    } else {
        None
    };

    if ief_debug_enabled() {
        // per-direction dS RMS and dSii_dF stats
        for xyz in 0..3 {
            let mut s = 0.0f64;
            let n = dS[xyz].size[0];
            for i in 0..n { for j in 0..n { s += dS[xyz][(i,j)].powi(2); } }
            let rms = (s / (n*n) as f64).sqrt();
            eprintln!("IEF_DBG| dD-dS dS[{}] rms={:.6e}", xyz, rms);
        }
        let dss: f64 = dSii_dF.iter().map(|v| v.abs()).sum::<f64>() / dSii_dF.len() as f64;
        eprintln!("IEF_DBG| dD-dS dSii_dF mean|abs|={:.6e}", dss);
    }

    (dD, dS, dSii_dF)
}

// ============================================================================
// grad_solvent_nuc
// ============================================================================

/// Nuclear potential contribution:  `q_sym · ∂v_nuc / ∂R`.
pub fn grad_solvent_nuc(
    surface: &SurfaceVdwGaussian,
    pscf: &PcmScf,
    mol: &Molecule,
    natm: usize,
) -> MatrixFull<f64> {
    let _t = Instant::now();
    let grid_coords = &surface.surface_calc.grid_coords;
    let charge_exp  = &surface.surface_calc.charge_exp;
    let exponents: Vec<f64> = charge_exp.iter().map(|&x| x * x).collect();
    let gslice = &surface.gslice_by_atom;
    let q_sym  = &pscf.q_sym;   // [ngrids, 1]
    let ngrids = grid_coords.len();

    let fake_grid = CINTR2CDATA::fakemol_for_charges(grid_coords, exponents.as_slice());
    let atm_pos: Vec<[f64; 3]> = (0..natm)
        .map(|i| [surface.atom_coords[(0,i)],
                  surface.atom_coords[(1,i)],
                  surface.atom_coords[(2,i)]])
        .collect();
    let fake_nuc  = CINTR2CDATA::fakemol_for_charges(&atm_pos, None);

    let mass_chg = get_mass_charge(&mol.geom.elem);
    let Z: Vec<f64> = mass_chg.iter().map(|&(_, c)| c as f64).collect();

    // ---- Part 1 : int2c2e_ip1(nuc, grid)   ∂/∂R_nuc ----
    let (raw1, sh1) =
        CINTR2CDATA::integrate_cross("int2c2e_ip1", [&fake_nuc, &fake_grid], None, None).into();
    // shape [natm, ngrids, 3], column-major: index = a + natm*g + natm*ngrids*t
    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);
    let n1: usize = sh1.iter().product();
    if n1 == natm * ngrids * 3 {
        for t in 0..3 {
            for g in 0..ngrids {
                let qv = q_sym[(g, 0)];
                if qv.abs() < 1e-15 { continue; }
                let base_tg = natm * (g + ngrids * t);
                for a in 0..natm {
                    de[(t, a)] -= raw1[a + base_tg] * Z[a] * qv;
                }
            }
        }
    } else {
        eprintln!("grad_solvent_nuc: unexpected int2c2e_ip1 shape {:?}, expected [natm, ngrids, 3]", sh1);
    }

    // ---- Part 2 : int2c2e_ip1(grid, nuc)   ∂/∂R_grid  (translational invariance) ----
    let (raw2, sh2) =
        CINTR2CDATA::integrate_cross("int2c2e_ip1", [&fake_grid, &fake_nuc],None, None).into();
    // shape [ngrids, natm, 3], column-major: index = g + ngrids*n + ngrids*natm*t
    let n2: usize = sh2.iter().product();
    if n2 == ngrids * natm * 3 {
        // dv_g[t, g] = Σ_n Z[n] * raw2[g, n, t]
        let mut dv_g = vec![0.0; ngrids * 3];
        for t in 0..3 {
            for n in 0..natm {
                let zn = Z[n];
                let base_tn = ngrids * (n + natm * t);
                for g in 0..ngrids {
                    dv_g[t * ngrids + g] += raw2[g + base_tn] * zn;
                }
            }
        }
        // de[t, a] -= Σ_{g∈gslice[a]} q_sym[g] * dv_g[t, g]
        for t in 0..3 {
            for a in 0..natm {
                let (p0, p1) = gslice[a];
                let mut s = 0.0;
                let base_t = t * ngrids;
                for g in p0..p1 {
                    s += q_sym[(g, 0)] * dv_g[base_t + g];
                }
                de[(t, a)] -= s;
            }
        }
    }

    de
}

// ============================================================================
// grad_solvent_qv
// ============================================================================

/// Electronic potential contribution:  `– q_sym · ∂v_e / ∂R`.
pub fn grad_solvent_qv(
    surface: &SurfaceVdwGaussian,
    pscf: &PcmScf,
    mol: &Molecule,
    dm_total: &MatrixFull<f64>,
    natm: usize,
    nao: usize,
) -> MatrixFull<f64> {
    let _t = Instant::now();
    let grid_coords = &surface.surface_calc.grid_coords;
    let charge_exp  = &surface.surface_calc.charge_exp;
    let exponents: Vec<f64> = charge_exp.iter().map(|&x| x * x).collect();
    let gslice = &surface.gslice_by_atom;
    let q_sym  = &pscf.q_sym;
    let ngrids = grid_coords.len();
    let aoslice = mol.aoslice_by_atom();

    let mut cint = mol.initialize_cint(false);

    // Chunk size: keep ~500 MB per batch
    let chunk = ((500_000_000.0 / (3.0 * nao as f64 * nao as f64 * 8.0)) as usize)
        .max(16).min(ngrids);

    // ---- Part A : int3c2e_ip1  (derivative w.r.t. AO centres μ, ν) ----
    // dvj_ao[(t, i)] = ∫ Σ_{μ,ν} D_{μν} · ∂(μν|g)/∂R_i[t]
    let mut dvj_ao = MatrixFull::<f64>::new([3, nao], 0.0);

    let mut p0 = 0;
    while p0 < ngrids {
        let p1 = (p0 + chunk).min(ngrids);
        let nc = p1 - p0;
        let grid_coords_chunk = &grid_coords[p0..p1];
        let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
        for &x in &charge_exp[p0..p1] {
            charge_exp_chunk.push(x * x);
        }
        let fake = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

        let (raw, sh) =
            CINTR2CDATA::integrate_cross("int3c2e_ip1", [&cint, &cint, &fake], None, None).into();
        // shape [nao, nao, nc, 3], column-major: index = i + nao*j + nao*nao*gi + nao*nao*nc*t
        let np: usize = sh.iter().product();
        if np == nao * nao * nc * 3 && sh.len() >= 3 {
            let n3 = sh.last().copied().unwrap_or(3) as usize;
            if n3 == 3 {
                for t in 0..3 {
                    for gi in 0..nc {
                        let g_idx = p0 + gi;
                        let qv = q_sym[(g_idx, 0)];
                        if qv.abs() < 1e-15 { continue; }
                        let base_tgi = nao * nao * (gi + nc * t);
                        for j in 0..nao {
                            let base_j = base_tgi + j * nao;
                            for i in 0..nao {
                                let dmv = dm_total[(i, j)];
                                if dmv.abs() < 1e-15 { continue; }
                                dvj_ao[(t, i)] += raw[base_j + i] * dmv * qv;
                            }
                        }
                    }
                }
            }
        }
        p0 = p1;
    }

    // Reduce AO → atoms ;  factor 2 for μ+ν symmetry
    let mut de_ip1 = MatrixFull::<f64>::new([3, natm], 0.0);
    for a in 0..natm {
        let [_, _, ao0, ao1] = aoslice[a];
        for t in 0..3 {
            let mut s = 0.0;
            for i in ao0..ao1 { s += dvj_ao[(t, i)]; }
            de_ip1[(t, a)] = 2.0 * s;
        }
    }

    // ---- Part B : int3c2e_ip2  (derivative w.r.t. grid centres) ----
    // dq_grid[(g, t)] = Σ_{μ,ν} D_{μν} · ∂(μν|g)/∂R_g[t] · q_sym[g]
    let mut dq_grid = MatrixFull::<f64>::new([ngrids, 3], 0.0);

    p0 = 0;
    while p0 < ngrids {
        let p1 = (p0 + chunk).min(ngrids);
        let nc = p1 - p0;
        let grid_coords_chunk = &grid_coords[p0..p1];
        let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
        for &x in &charge_exp[p0..p1] {
            charge_exp_chunk.push(x * x);
        }
        let fake = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

        let (raw, sh) =
            CINTR2CDATA::integrate_cross("int3c2e_ip2",
                                          [&cint, &cint, &fake], None, None).into();
        // shape [nao, nao, nc, 3], column-major: index = i + nao*j + nao*nao*gi + nao*nao*nc*t
        let np: usize = sh.iter().product();
        if np == nao * nao * nc * 3 && sh.len() >= 3 {
            let n3 = sh.last().copied().unwrap_or(3) as usize;
            if n3 == 3 {
                for t in 0..3 {
                    for gi in 0..nc {
                        let g_idx = p0 + gi;
                        let qv = q_sym[(g_idx, 0)];
                        let base_tgi = nao * nao * (gi + nc * t);
                        for j in 0..nao {
                            let base_j = base_tgi + j * nao;
                            for i in 0..nao {
                                let dmv = dm_total[(i, j)];
                                if dmv.abs() < 1e-15 { continue; }
                                dq_grid[(g_idx, t)] += raw[base_j + i] * dmv * qv;
                            }
                        }
                    }
                }
            }
        }
        p0 = p1;
    }

    // Reduce grid → atoms
    let mut de_ip2 = MatrixFull::<f64>::new([3, natm], 0.0);
    for a in 0..natm {
        let (p0, p1) = gslice[a];
        for t in 0..3 {
            let mut s = 0.0;
            for g in p0..p1 { s += dq_grid[(g, t)]; }
            de_ip2[(t, a)] = s;
        }
    }

    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);
    for a in 0..natm { for t in 0..3 { de[(t, a)] = de_ip1[(t, a)] + de_ip2[(t, a)]; } }
    de
}

// ============================================================================
// SolverAux — pre‑computed intermediate vectors (shared by all de_* fns)
// ============================================================================

/// Pre‑computed vectors for the solver gradient decomposition.
///
/// Each vector is computed once and reused across all `compute_de_*` calls.
struct SolverAux {
    /// vk1 = K^{-T} · v_grids,  [ngrids]
    vk1: Vec<f64>,
    /// Sq = S · q_sym,           [ngrids, 1]  (column vector)
    sq: MatrixFull<f64>,
    /// vk1^T · D,                [1, ngrids]  (row vector)
    vk1_d: MatrixFull<f64>,
    /// vk1^T · D · A  (element‑wise: vk1_d[i] · A[i]),  [ngrids]
    vk1_da: Vec<f64>,
    /// vk1^T · S,                [ngrids]  (SSVPE only, zeroed otherwise)
    vk1_s: Vec<f64>,
    /// D^T · q_sym,              [ngrids]  (SSVPE only, zeroed otherwise)
    dt_q: Vec<f64>,
}

/// Pre‑compute all auxiliary vectors needed for solver gradient decomposition.
///
/// # Input
/// - `pstatic`: contains S, D, A matrices
/// - `v_grids`: electrostatic potential on surface grids \[ngrids\]
/// - `q_sym`:   symmetrized apparent surface charges \[ngrids, 1\]
/// - `method`:  PCM method (controls which vectors to compute)
///
/// Ref: `grad_solver_derivation.md` §11 预计算向量汇总
fn compute_solver_aux(
    pstatic: &PcmStatic,
    v_grids: &[f64],
    q_sym: &MatrixFull<f64>,
    method: &PcmMethod,
) -> SolverAux {
    let ngrids = v_grids.len();
    let K = &pstatic.K;
    let K_ipiv = &pstatic.K_ipiv;

    // vk1 = K^{-T} · v_grids
    let vk1 = solve_lu_transpose(K, K_ipiv, v_grids)
        .expect("compute_solver_aux: solve_lu_transpose failed");

    // Sq = S · q_sym   (BLAS: S[n,n] × q_sym[n,1] → [n,1])
    let sq = _dgemm_scaled(&pstatic.S, 'N', q_sym, 'N', 1.0);

    // vk1_d = vk1^T · D   (BLAS: vk1[1,n]^T × D[n,n] → [1,n])
    // Vec<f64> has BasicMatrix impl as [n,1] column; 'T' transposes to [1,n] row
    let vk1_d = _dgemm_scaled(&vk1, 'T', &pstatic.D, 'N', 1.0);

    // vk1_da = vk1_d ⊙ A   (element‑wise)
    let vk1_da: Vec<f64> = (0..ngrids)
        .map(|i| vk1_d[(0, i)] * pstatic.A[i])
        .collect();

    let is_ssvpe = matches!(method, PcmMethod::SSVPE);

    // vk1_s = vk1^T · S   (SSVPE only)
    let vk1_s: Vec<f64> = if is_ssvpe {
        let vk1_s_mat = _dgemm_scaled(&vk1, 'T', &pstatic.S, 'N', 1.0);
        (0..ngrids).map(|i| vk1_s_mat[(0, i)]).collect()
    } else {
        vec![0.0; ngrids]
    };

    // dt_q = D^T · q_sym   (SSVPE only)
    let dt_q: Vec<f64> = if is_ssvpe {
        let dt_q_mat = _dgemm_scaled(&pstatic.D, 'T', q_sym, 'N', 1.0);
        (0..ngrids).map(|i| dt_q_mat[(i, 0)]).collect()
    } else {
        vec![0.0; ngrids]
    };

    SolverAux { vk1, sq, vk1_d, vk1_da, vk1_s, dt_q }
}

// ============================================================================
// Private helpers — reusable across compute_de_* functions
// ============================================================================

/// Antisymmetrised chain rule:  u^T · dM · w  →  [3, natm].
///
/// For each atom A and direction xyz ∈ {0,1,2}:
/// ```text
///   de[A, xyz] = Σ_{i∈A} u[i] · (dM_xyz · w)[i]
///              − Σ_{j∈A} w[j] · (dM_xyz^T · u)[j]
/// ```
///
/// The derivative matrices satisfy translational invariance:
/// ```text
///   dM[xyz][(i,j)] = ∂M_ij/∂r_i[xyz]
///   ∂M_ij/∂r_j[xyz] = −∂M_ij/∂r_i[xyz]
/// ```
///
/// # Arguments
/// - `u`:    left vector         [ngrids]
/// - `dm`:   dM/dr_i per xyz     [3] of [ngrids, ngrids]
/// - `w`:    right vector        [ngrids]
///
/// # Returns
/// `de` shape [3, natm] — raw geometric contribution, **no prefactor**.
///
/// Ref: `grad_solver_derivation.md` §5 链式法则
fn antisym_chain_rule(
    u: &[f64],
    dm: &[MatrixFull<f64>],
    w: &[f64],
    gslice: &[(usize, usize)],
    natm: usize,
) -> MatrixFull<f64> {
    let ngrids = u.len();
    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);

    for xyz in 0..3 {
        // dm_w = dM[xyz] · w   (BLAS gemv: [n,n] × [n,1] → [n,1])
        let w_col: Vec<f64> = w.to_vec();
        let dm_w = _dgemm_scaled(&dm[xyz], 'N', &w_col, 'N', 1.0);

        // dmt_u = dM[xyz]^T · u   (BLAS gemv: [n,n]^T × [n,1] → [n,1])
        let u_col: Vec<f64> = u.to_vec();
        let dmt_u = _dgemm_scaled(&dm[xyz], 'T', &u_col, 'N', 1.0);

        for a in 0..natm {
            let (p0, p1) = gslice[a];
            let mut s = 0.0;

            // grid response:  Σ_{i∈A} u[i] · dm_w[i]
            for i in p0..p1 { s += u[i] * dm_w[(i, 0)]; }

            // atom response:  −Σ_{j∈A} w[j] · dmt_u[j]
            for j in p0..p1 { s -= w[j] * dmt_u[(j, 0)]; }

            de[(xyz, a)] = s;
        }
    }

    de
}

/// Diagonal S_ii correction:  Σ_i (u_i·q_i) · (∂S_ii/∂F_i) · (∂F_i/∂R_A).
///
/// ```text
///   de[A, xyz] = Σ_i (u[i]·q[i]) · dSii_dF[i] · dF[(i, a*3+xyz)]
/// ```
///
/// # Arguments
/// - `u_mul_q`: u[i] * q[i]           [ngrids]
/// - `dsii_df`: ∂S_ii/∂F_i            [ngrids]
/// - `df`:      ∂F_i/∂R_a[xyz]        [ngrids, natm*3]
///
/// # Returns
/// `de` shape [3, natm] — raw geometric contribution, **no prefactor**.
///
/// Ref: `grad_solver_derivation.md` §7 对角元修正
fn diag_s_correction(
    u_mul_q: &[f64],
    dsii_df: &[f64],
    df: &MatrixFull<f64>,
    natm: usize,
) -> MatrixFull<f64> {
    let ngrids = u_mul_q.len();
    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);
    for a in 0..natm {
        for xyz in 0..3 {
            let mut s = 0.0;
            for i in 0..ngrids {
                s += u_mul_q[i] * dsii_df[i] * df[(i, a * 3 + xyz)];
            }
            de[(xyz, a)] = s;
        }
    }
    de
}

/// dA contraction:  Σ_i w[i] · ∂A_i/∂R_A.
///
/// A is a diagonal matrix → no off‑diagonal chain rule.
///
/// ```text
///   de[A, xyz] = Σ_i w[i] · dA[(i, a*3+xyz)]
/// ```
///
/// # Arguments
/// - `w`:    weight vector   [ngrids]
/// - `da`:   ∂A_i/∂R_a[xyz]  [ngrids, natm*3]
///
/// # Returns
/// `de` shape [3, natm] — raw geometric contribution, **no prefactor**.
///
/// Ref: `grad_solver_derivation.md` §9.3 项 3
fn da_contract(
    w: &[f64],
    da: &MatrixFull<f64>,
    natm: usize,
) -> MatrixFull<f64> {
    let ngrids = w.len();
    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);
    for a in 0..natm {
        for xyz in 0..3 {
            let mut s = 0.0;
            for i in 0..ngrids {
                s += w[i] * da[(i, a * 3 + xyz)];
            }
            de[(xyz, a)] = s;
        }
    }
    de
}

// ============================================================================
// Public gradient contribution functions (prefactor‑free)
//
// Each function returns the raw geometric contribution (no ½, no α/γ).
// Prefactors are applied by the caller in grad_solvent_solver.
//
// Ref: design spec §5, derivation §9–§10
// ============================================================================

/// `de_dS0`: pure S‑matrix derivative contribution.
///
/// ```text
///   de_dS0 = antisym_chain_rule(vk1, dS, q) + diag_s_correction(vk1⊙q, dSii_dF, dF)
/// ```
///
/// Used by all PCM methods. Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §9.3 项 1
pub fn compute_de_ds0(
    vk1: &[f64],
    q_sym: &MatrixFull<f64>,
    ds: &[MatrixFull<f64>],
    dsii_df: &[f64],
    df: &MatrixFull<f64>,
    gslice: &[(usize, usize)],
    natm: usize,
) -> MatrixFull<f64> {
    let ngrids = vk1.len();

    // off‑diagonal
    let q_flat: Vec<f64> = (0..ngrids).map(|i| q_sym[(i, 0)]).collect();
    let mut de = antisym_chain_rule(vk1, ds, &q_flat, gslice, natm);
    if ief_debug_enabled() { ief_print_grad("de-s0.antisym", &de); }

    // diagonal correction: vk1⊙q
    let vk1_mul_q: Vec<f64> = vk1.iter()
        .zip(q_flat.iter())
        .map(|(v, q)| v * q)
        .collect();
    let de_diag = diag_s_correction(&vk1_mul_q, dsii_df, df, natm);
    if ief_debug_enabled() { ief_print_grad("de-s0.diag", &de_diag); }

    for i in 0..de.data.len() { de.data[i] += de_diag.data[i]; }
    de
}

/// `de_dD`: D‑matrix derivative contribution.
///
/// ```text
///   de_dD = antisym_chain_rule(vk1, dD, w)
/// ```
///
/// `w` = A⊙v (dR part) or A⊙Sq (dK part). Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §9.3 项 2
pub fn compute_de_dd(
    vk1: &[f64],
    dd: &[MatrixFull<f64>],
    w: &[f64],
    gslice: &[(usize, usize)],
    natm: usize,
) -> MatrixFull<f64> {
    antisym_chain_rule(vk1, dd, w, gslice, natm)
}

/// `de_dA`: area diagonal‑matrix derivative contribution.
///
/// ```text
///   de_dA = da_contract(weight, dA)
/// ```
///
/// `weight` = vk1_D⊙v (dR part) or vk1_D⊙Sq (dK part). Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §9.3 项 3
pub fn compute_de_da(
    weight: &[f64],
    da: &MatrixFull<f64>,
    natm: usize,
) -> MatrixFull<f64> {
    da_contract(weight, da, natm)
}

/// `de_dS1`: D·A·dS correction term.
///
/// ```text
///   de_dS1 = antisym_chain_rule(vk1_da, dS, q) + diag_s_correction(vk1_da⊙q, dSii_dF, dF)
/// ```
///
/// vk1_da = vk1^T·D·A  (pre‑computed in SolverAux). Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §9.3 项 4
pub fn compute_de_ds1(
    vk1_da: &[f64],
    q_sym: &MatrixFull<f64>,
    ds: &[MatrixFull<f64>],
    dsii_df: &[f64],
    df: &MatrixFull<f64>,
    gslice: &[(usize, usize)],
    natm: usize,
) -> MatrixFull<f64> {
    let ngrids = vk1_da.len();

    // off‑diagonal
    let q_flat: Vec<f64> = (0..ngrids).map(|i| q_sym[(i, 0)]).collect();
    let mut de = antisym_chain_rule(vk1_da, ds, &q_flat, gslice, natm);

    // diagonal: vk1_da⊙q
    let vk1_da_mul_q: Vec<f64> = vk1_da.iter()
        .zip(q_flat.iter())
        .map(|(d, q)| d * q)
        .collect();
    let de_diag = diag_s_correction(&vk1_da_mul_q, dsii_df, df, natm);

    for i in 0..de.data.len() { de.data[i] += de_diag.data[i]; }
    de
}

/// `de_dS1_T`: dS·A·D^T term (SSVPE only).
///
/// ```text
///   de_dS1_T = antisym_chain_rule(vk1, dS, ADT_q)
///            + diag_s_correction(vk1⊙ADT_q, dSii_dF, dF)
/// ```
///
/// ADT_q = A ⊙ (D^T·q). Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §10.3 de_dS1_T
pub fn compute_de_ds1_t(
    vk1: &[f64],
    adt_q: &[f64],
    ds: &[MatrixFull<f64>],
    dsii_df: &[f64],
    df: &MatrixFull<f64>,
    gslice: &[(usize, usize)],
    natm: usize,
) -> MatrixFull<f64> {
    // off‑diagonal
    let mut de = antisym_chain_rule(vk1, ds, adt_q, gslice, natm);

    // diagonal: vk1⊙ADT_q
    let vk1_mul_adtq: Vec<f64> = vk1.iter()
        .zip(adt_q.iter())
        .map(|(v, a)| v * a)
        .collect();
    let de_diag = diag_s_correction(&vk1_mul_adtq, dsii_df, df, natm);

    for i in 0..de.data.len() { de.data[i] += de_diag.data[i]; }
    de
}

/// `de_dD_T`: S·A·dD^T term (SSVPE only).
///
/// Uses the identity  u^T·(−dD^T)·w  expanded via the antisym chain rule:
/// ```text
///   de[A,xyz] = −Σ_{i∈A} u[i] · (dD^T·w)[i]  +  Σ_{j∈A} w[j] · (dD·u)[j]
/// ```
/// This avoids explicitly constructing the transposed dD matrix.
///
/// vk1_sa = vk1^T·S·A. Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §10.3 de_dD_T
pub fn compute_de_dd_t(
    vk1_sa: &[f64],
    q_sym: &MatrixFull<f64>,
    dd: &[MatrixFull<f64>],
    gslice: &[(usize, usize)],
    natm: usize,
) -> MatrixFull<f64> {
    let ngrids = vk1_sa.len();
    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);

    for xyz in 0..3 {
        // dD^T · q   (BLAS: dD^T[n,n] × q[n,1] → [n,1])
        let ddt_q = _dgemm_scaled(&dd[xyz], 'T', q_sym, 'N', 1.0);

        // dD · vk1_sa   (BLAS: dD[n,n] × vk1_sa[n,1] → [n,1])
        let vk1_sa_col: Vec<f64> = vk1_sa.to_vec();
        let dd_u = _dgemm_scaled(&dd[xyz], 'N', &vk1_sa_col, 'N', 1.0);

        for a in 0..natm {
            let (p0, p1) = gslice[a];
            let mut s = 0.0;

            // grid response:  −Σ_{i∈A} vk1_sa[i] · (dD^T·q)[i]
            for i in p0..p1 { s -= vk1_sa[i] * ddt_q[(i, 0)]; }

            // atom response:  +Σ_{j∈A} q[j] · (dD·vk1_sa)[j]
            for j in p0..p1 { s += q_sym[(j, 0)] * dd_u[(j, 0)]; }

            de[(xyz, a)] = s;
        }
    }

    de
}

/// `de_dA_T`: S·dA·D^T term (SSVPE only).
///
/// ```text
///   de_dA_T = da_contract(vk1_S ⊙ DT_q, dA)
/// ```
///
/// vk1_S = vk1^T·S,  DT_q = D^T·q  (pre‑computed in SolverAux).
/// Caller applies ½ prefactor.
///
/// Ref: `grad_solver_derivation.md` §10.3 de_dA_T
pub fn compute_de_da_t(
    weight: &[f64],
    da: &MatrixFull<f64>,
    natm: usize,
) -> MatrixFull<f64> {
    da_contract(weight, da, natm)
}

// ============================================================================
// grad_solvent_solver  (cavity response)
// ============================================================================

/// Cavity‑response contribution : `½ v^T · d(K⁻¹R)/dx · v`.
///
/// For CPCM / COSMO:
/// ```text
///   dE_solver/dR_A[xyz] = ½ vK_1^T · dS/dR_A · q_sym
///                       = ½ [ Σ_{i∈A} vK_1[i] · (dS·q_sym)[i]
///                           − Σ_{j∈A} q_sym[j] · (dS^T·vK_1)[j]
///                           + Σ_{i∈A} vK_1[i] · q_sym[i] · dSii_xyz[i] ]
/// ```
///
/// The off‑diagonal (dS·q_sym, dS^T·vK_1) is pre‑computed via BLAS for each xyz
/// direction, removing the O(ngrids × patch_size) double loops.
pub fn grad_solvent_solver(
    surface: &SurfaceVdwGaussian,
    pstatic: &PcmStatic,
    pscf: &PcmScf,
    method: &PcmMethod,
    natm: usize,
) -> MatrixFull<f64> {
    let _t = Instant::now();
    let gslice   = &surface.gslice_by_atom;
    let v_grids  = &pscf.v_grids;            // [ngrids]
    let q_sym    = &pscf.q_sym;              // [ngrids, 1]
    let f_eps    = pstatic.f_epsilon;

    // ---- pre‑compute auxiliary vectors ----
    let aux = compute_solver_aux(pstatic, v_grids, q_sym, method);
    if ief_debug_enabled() {
        let ng = v_grids.len();
        let vk1_rms = (aux.vk1.iter().map(|x| x*x).sum::<f64>() / ng as f64).sqrt();
        let q_rms = (0..ng).map(|i| q_sym[(i,0)]*q_sym[(i,0)]).sum::<f64>();
        let q_rms = (q_rms / ng as f64).sqrt();
        eprintln!("IEF_DBG| solver-aux vk1[0]={:.12e} vk1_rms={:.12e} q[0]={:.12e} q_rms={:.12e} f_eps={:.12e}",
            aux.vk1[0], vk1_rms, q_sym[(0,0)], q_rms, f_eps);
    }

    // ---- derivative matrices ----
    let (dF, dA_deriv) = get_dF_dA(surface);
    if ief_debug_enabled() {
        eprintln!("IEF_DBG| dF-dA dF.size={:?} dA.size={:?} dF[0,0]={:.12e} dA[0,0]={:.12e}",
            dF.size, dA_deriv.size, dF[(0,0)], dA_deriv[(0,0)]);
    }
    let with_D = matches!(method, PcmMethod::IEFPCM | PcmMethod::SSVPE | PcmMethod::SMD);
    let (dD_opt, dS, dSii_dF) = get_dD_dS(surface, &dF, with_D);
    if ief_debug_enabled() {
        eprintln!("IEF_DBG| dD-dS dS[0].size=({},{}) dSii_dF[0]={:.12e}",
            dS[0].size[0], dS[0].size[1], dSii_dF[0]);
        if let Some(ref dd) = dD_opt {
            eprintln!("IEF_DBG| dD-dS dD[0].size=({},{}) dD[0,(0,0)]={:.12e}",
                dd[0].size[0], dd[0].size[1], dd[0][(0,0)]);
        }
    }

    match method {
        // ================================================================
        // CPCM / COSMO:  K = S,  R = −f_ε·I  →  dR = 0,  dK = dS
        //   dE = −½ · vK1^T · dS · q
        // ================================================================
        PcmMethod::CPCM | PcmMethod::COSMO => {
            // de_s0: raw geometric = antisym(vk1, dS, q) + diag(vk1⊙q, dSii_dF, dF)
            let de_s0_raw = compute_de_ds0(&aux.vk1, q_sym, &dS, &dSii_dF, &dF, gslice, natm);
            if ief_debug_enabled() { ief_print_grad("assembly.de-s0", &de_s0_raw); }
            let mut de = de_s0_raw;
            // de = −½ · de_s0   (sign from dE = −½ vK1^T·dK·q with dK=dS)
            for v in de.data.iter_mut() { *v *= -0.5; }
            if ief_debug_enabled() { ief_print_grad("assembly.final", &de); }
            de
        }

        // ================================================================
        // IEFPCM / SMD
        //   K = S − α·D·A·S,   R = −f_ε·I + α·D·A,   α = f_ε/(2π)
        //   dR = α·(dD·A + D·dA)
        //   dK = dS − α·(dD·A·S + D·dA·S + D·A·dS)
        //   dE = +½ vK1^T·dR·v − ½ vK1^T·dK·q
        // ================================================================
        PcmMethod::IEFPCM | PcmMethod::SMD => {
            let alpha = f_eps / (2.0 * PI);
            let dD = dD_opt.as_ref().expect("IEFPCM: dD must be computed");
            let ngrids = aux.vk1.len();

            // --- de_dR ---
            // dR part·dD:  antisym(vk1, dD, A⊙v)
            let av: Vec<f64> = pstatic.A.iter()
                .zip(v_grids.iter())
                .map(|(a, v)| a * v)
                .collect();
            let mut de_r = compute_de_dd(&aux.vk1, dD, &av, gslice, natm);

            // dR part·dA:  da_contract(vk1_D⊙v, dA)
            let vk1_d_mul_v: Vec<f64> = (0..ngrids)
                .map(|i| aux.vk1_d[(0, i)] * v_grids[i])
                .collect();
            let de_r_da = compute_de_da(&vk1_d_mul_v, &dA_deriv, natm);
            for i in 0..de_r.data.len() { de_r.data[i] += de_r_da.data[i]; }
            // prefactor:  ½α
            for v in de_r.data.iter_mut() { *v *= 0.5 * alpha; }
            if ief_debug_enabled() { ief_print_grad("assembly.de-r", &de_r); }

            // --- de_dK (to be subtracted) ---
            // de_s0: ½ · [antisym(vk1, dS, q) + diag(vk1⊙q, dSii_dF, dF)]
            let de_s0_raw = compute_de_ds0(&aux.vk1, q_sym, &dS, &dSii_dF, &dF, gslice, natm);
            if ief_debug_enabled() { ief_print_grad("assembly.de-s0", &de_s0_raw); }

            // de_dD: ½ · antisym(vk1, dD, A⊙Sq)
            let asq: Vec<f64> = pstatic.A.iter()
                .zip((0..ngrids).map(|i| aux.sq[(i, 0)]))
                .map(|(a, s)| a * s)
                .collect();
            let mut de_dd = compute_de_dd(&aux.vk1, dD, &asq, gslice, natm);
            for v in de_dd.data.iter_mut() { *v *= 0.5; }
            if ief_debug_enabled() { ief_print_grad("assembly.de-dd", &de_dd); }

            // de_dA: ½ · da_contract(vk1_D⊙Sq, dA)
            let vk1_d_mul_sq: Vec<f64> = (0..ngrids)
                .map(|i| aux.vk1_d[(0, i)] * aux.sq[(i, 0)])
                .collect();
            let mut de_da = compute_de_da(&vk1_d_mul_sq, &dA_deriv, natm);
            for v in de_da.data.iter_mut() { *v *= 0.5; }
            if ief_debug_enabled() { ief_print_grad("assembly.de-da", &de_da); }

            // de_dS1: ½ · [antisym(vk1_da, dS, q) + diag(vk1_da⊙q, dSii_dF, dF)]
            let mut de_s1 = compute_de_ds1(&aux.vk1_da, q_sym, &dS, &dSii_dF, &dF, gslice, natm);
            for v in de_s1.data.iter_mut() { *v *= 0.5; }
            if ief_debug_enabled() { ief_print_grad("assembly.de-s1", &de_s1); }

            // de = de_r − de_s0 + α·(de_dd + de_da + de_s1)
            let mut de = de_r.clone();
            for i in 0..de.data.len() {
                de.data[i] -= 0.5 * de_s0_raw.data[i]
                    - alpha * (de_dd.data[i] + de_da.data[i] + de_s1.data[i]);
            }
            if ief_debug_enabled() { ief_print_grad("assembly.final", &de); }

            // DEBUG: print per‑component stats for PySCF comparison
            {
                // scale components to their final values for display
                let mut de_s0_final = de_s0_raw;
                for v in de_s0_final.data.iter_mut() { *v *= -0.5; }
                let (s0min, s0max, s0rms) = grad_stats(&de_s0_final);
                let (ddmin, ddmax, ddrms) = grad_stats(&de_dd);
                let (damin, damax, darms) = grad_stats(&de_da);
                let (s1min, s1max, s1rms) = grad_stats(&de_s1);
                let (rmin, rmax, rrms) = grad_stats(&de_r);
                let (tmin, tmax, trms) = grad_stats(&de);
                eprintln!("[IEFPCM grad components] de_r(dR): (min={rmin:.6e}, max={rmax:.6e}, rms={rrms:.6e})");
                eprintln!("[IEFPCM grad components] de_s0:   (min={s0min:.6e}, max={s0max:.6e}, rms={s0rms:.6e})");
                eprintln!("[IEFPCM grad components] de_dD:   (min={ddmin:.6e}, max={ddmax:.6e}, rms={ddrms:.6e})");
                eprintln!("[IEFPCM grad components] de_dA:   (min={damin:.6e}, max={damax:.6e}, rms={darms:.6e})");
                eprintln!("[IEFPCM grad components] de_dS1:  (min={s1min:.6e}, max={s1max:.6e}, rms={s1rms:.6e})");
                eprintln!("[IEFPCM grad components] TOTAL:   (min={tmin:.6e}, max={tmax:.6e}, rms={trms:.6e})");
            }

            de
        }

        // ================================================================
        // SSVPE
        //   K = S − γ·(D·A·S + S·A·D^T),   γ = f_ε/(4π) = α/2
        //   R: same as IEFPCM → dR same
        //   dK = dS − γ·(dD·A·S + D·dA·S + D·A·dS
        //               + dS·A·D^T + S·dA·D^T + S·A·dD^T)
        // ================================================================
        PcmMethod::SSVPE => {
            let alpha = f_eps / (2.0 * PI);   // for dR
            let gamma = f_eps / (4.0 * PI);   // for dK  (= α/2)
            let dD = dD_opt.as_ref().expect("SSVPE: dD must be computed");
            let ngrids = aux.vk1.len();

            // --- de_dR: same as IEFPCM ---
            let av: Vec<f64> = pstatic.A.iter()
                .zip(v_grids.iter())
                .map(|(a, v)| a * v)
                .collect();
            let mut de_r = compute_de_dd(&aux.vk1, dD, &av, gslice, natm);
            let vk1_d_mul_v: Vec<f64> = (0..ngrids)
                .map(|i| aux.vk1_d[(0, i)] * v_grids[i])
                .collect();
            let de_r_da = compute_de_da(&vk1_d_mul_v, &dA_deriv, natm);
            for i in 0..de_r.data.len() { de_r.data[i] += de_r_da.data[i]; }
            for v in de_r.data.iter_mut() { *v *= 0.5 * alpha; }

            // --- de_s0 (same as CPCM) ---
            let de_s0_raw = compute_de_ds0(&aux.vk1, q_sym, &dS, &dSii_dF, &dF, gslice, natm);

            // --- de_dD, de_dA, de_dS1 (same as IEFPCM) ---
            let asq: Vec<f64> = pstatic.A.iter()
                .zip((0..ngrids).map(|i| aux.sq[(i, 0)]))
                .map(|(a, s)| a * s)
                .collect();
            let mut de_dd = compute_de_dd(&aux.vk1, dD, &asq, gslice, natm);
            for v in de_dd.data.iter_mut() { *v *= 0.5; }

            let vk1_d_mul_sq: Vec<f64> = (0..ngrids)
                .map(|i| aux.vk1_d[(0, i)] * aux.sq[(i, 0)])
                .collect();
            let mut de_da = compute_de_da(&vk1_d_mul_sq, &dA_deriv, natm);
            for v in de_da.data.iter_mut() { *v *= 0.5; }

            let mut de_s1 = compute_de_ds1(&aux.vk1_da, q_sym, &dS, &dSii_dF, &dF, gslice, natm);
            for v in de_s1.data.iter_mut() { *v *= 0.5; }

            // --- SSVPE transpose terms ---
            // ADT_q = A ⊙ (D^T·q)
            let adt_q: Vec<f64> = pstatic.A.iter()
                .zip(aux.dt_q.iter())
                .map(|(a, d)| a * d)
                .collect();
            let mut de_s1_t = compute_de_ds1_t(&aux.vk1, &adt_q, &dS, &dSii_dF, &dF, gslice, natm);
            for v in de_s1_t.data.iter_mut() { *v *= 0.5; }

            // vk1_sa = vk1^T·S ⊙ A
            let vk1_sa: Vec<f64> = aux.vk1_s.iter()
                .zip(pstatic.A.iter())
                .map(|(s, a)| s * a)
                .collect();
            let mut de_dd_t = compute_de_dd_t(&vk1_sa, q_sym, dD, gslice, natm);
            for v in de_dd_t.data.iter_mut() { *v *= 0.5; }

            // vk1_S ⊙ DT_q
            let vk1_s_mul_dtq: Vec<f64> = aux.vk1_s.iter()
                .zip(aux.dt_q.iter())
                .map(|(s, d)| s * d)
                .collect();
            let mut de_da_t = compute_de_da_t(&vk1_s_mul_dtq, &dA_deriv, natm);
            for v in de_da_t.data.iter_mut() { *v *= 0.5; }

            // de = de_r − ½·de_s0_raw + γ·(de_dd + de_da + de_s1
            //                               + de_dd_t + de_da_t + de_s1_t)
            let mut de = de_r.clone();
            for i in 0..de.data.len() {
                de.data[i] -= 0.5 * de_s0_raw.data[i]
                    - gamma * (de_dd.data[i] + de_da.data[i] + de_s1.data[i]
                             + de_dd_t.data[i] + de_da_t.data[i] + de_s1_t.data[i]);
            }

            // DEBUG: print per‑component stats for PySCF comparison
            {
                let mut de_s0_final = de_s0_raw;
                for v in de_s0_final.data.iter_mut() { *v *= -0.5; }
                let (s0min, s0max, s0rms) = grad_stats(&de_s0_final);
                let (ddmin, ddmax, ddrms) = grad_stats(&de_dd);
                let (damin, damax, darms) = grad_stats(&de_da);
                let (s1min, s1max, s1rms) = grad_stats(&de_s1);
                let (s1tmin, s1tmax, s1trms) = grad_stats(&de_s1_t);
                let (ddtmin, ddtmax, ddtrms) = grad_stats(&de_dd_t);
                let (datmin, datmax, datrms) = grad_stats(&de_da_t);
                let (rmin, rmax, rrms) = grad_stats(&de_r);
                let (tmin, tmax, trms) = grad_stats(&de);
                eprintln!("[SSVPE grad components] de_r(dR):    (min={rmin:.6e}, max={rmax:.6e}, rms={rrms:.6e})");
                eprintln!("[SSVPE grad components] de_s0:      (min={s0min:.6e}, max={s0max:.6e}, rms={s0rms:.6e})");
                eprintln!("[SSVPE grad components] de_dD:      (min={ddmin:.6e}, max={ddmax:.6e}, rms={ddrms:.6e})");
                eprintln!("[SSVPE grad components] de_dA:      (min={damin:.6e}, max={damax:.6e}, rms={darms:.6e})");
                eprintln!("[SSVPE grad components] de_dS1:     (min={s1min:.6e}, max={s1max:.6e}, rms={s1rms:.6e})");
                eprintln!("[SSVPE grad components] de_dS1_T:   (min={s1tmin:.6e}, max={s1tmax:.6e}, rms={s1trms:.6e})");
                eprintln!("[SSVPE grad components] de_dD_T:    (min={ddtmin:.6e}, max={ddtmax:.6e}, rms={ddtrms:.6e})");
                eprintln!("[SSVPE grad components] de_dA_T:    (min={datmin:.6e}, max={datmax:.6e}, rms={datrms:.6e})");
                eprintln!("[SSVPE grad components] TOTAL:      (min={tmin:.6e}, max={tmax:.6e}, rms={trms:.6e})");
            }

            de
        }
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Helper: compute min/max/rms stats for a [3, natm] gradient matrix.
fn grad_stats(g: &MatrixFull<f64>) -> (f64, f64, f64) {
    let n = g.data.len();
    if n == 0 { return (0.0, 0.0, 0.0); }
    let mut gmin = g.data[0];
    let mut gmax = g.data[0];
    let mut ssq = 0.0;
    for &v in &g.data {
        if v < gmin { gmin = v; }
        if v > gmax { gmax = v; }
        ssq += v * v;
    }
    (gmin, gmax, (ssq / n as f64).sqrt())
}

/// Formatted table of a [3, natm] PCM gradient component w.r.t. atoms.
fn format_grad_component(label: &str, g: &MatrixFull<f64>, elem: &[String]) -> String {
    let (gmin, gmax, grms) = grad_stats(g);
    let mut out = format!("--- {label} ---  (min={gmin:12.6e}, max={gmax:12.6e}, rms={grms:12.6e})\n");
    for a in 0..elem.len() {
        out += &format!("  {:>3}{:16.8}{:16.8}{:16.8}\n",
            elem[a], g[(0, a)], g[(1, a)], g[(2, a)]);
    }
    out
}

/// Compute the full PCM solvent gradient = grad_nuc + grad_qv + grad_solver.
///
/// Returns `[3, natm]` force contribution from the implicit solvent model.
pub fn compute_solvent_gradient(scf_data: &SCF) -> MatrixFull<f64> {
    let t0 = Instant::now();
    let pcm_obj = scf_data.solvent_static_obj.as_ref()
        .expect("compute_solvent_gradient: solvent_static_obj missing");
    let pcm_scf = scf_data.solvent_scf.as_ref()
        .expect("compute_solvent_gradient: solvent_scf missing (run SCF first)");

    let surface = &pcm_obj.surface;
    let pstatic = &pcm_obj.pstatic;
    let pscf    = pcm_scf;
    let natm    = surface.atomic_num.len();
    let method  = &pcm_obj.cfg.method;
    let cint    = scf_data.mol.initialize_cint(false);
    let nao     = cint.nao();
    let elem    = &scf_data.mol.geom.elem;

    // Total density matrix (sum over spin channels)
    let dm = &scf_data.density_matrix;
    let mut dm_vec = vec![0.0; nao * nao];
    for i_spin in 0..scf_data.mol.spin_channel {
        for j in 0..nao {
            for i in 0..nao {
                dm_vec[j*nao + i] += dm[i_spin][(i,j)];
            }
        }
    }
    let dm_tot = MatrixFull::from_vec([nao, nao], dm_vec).unwrap();

    let print_lvl = scf_data.mol.ctrl.print_level;

    println!("--- PCM gradient components ---");

    let de_nuc    = grad_solvent_nuc(surface, pscf, &scf_data.mol, natm);
    let de_qv     = grad_solvent_qv(surface, pscf, &scf_data.mol, &dm_tot, natm, nao);
    let de_solver = grad_solvent_solver(surface, pstatic, pscf, method, natm);

    println!("  grad_nuc    done  (min={:12.6e}, max={:12.6e}, rms={:12.6e})",
        grad_stats(&de_nuc).0, grad_stats(&de_nuc).1, grad_stats(&de_nuc).2);
    println!("  grad_qv     done  (min={:12.6e}, max={:12.6e}, rms={:12.6e})",
        grad_stats(&de_qv).0, grad_stats(&de_qv).1, grad_stats(&de_qv).2);
    println!("  grad_solver done  (min={:12.6e}, max={:12.6e}, rms={:12.6e})",
        grad_stats(&de_solver).0, grad_stats(&de_solver).1, grad_stats(&de_solver).2);

    if print_lvl >= 2 {
        println!("{}", format_grad_component("de_solvent.nuc",    &de_nuc,    elem));
        println!("{}", format_grad_component("de_solvent.qv",     &de_qv,     elem));
        println!("{}", format_grad_component("de_solvent.solver", &de_solver, elem));
    }

    // ---- SMD CDS gradient (from cache) ----
    let de_cds = if let Some(ref d_cds) = pstatic.d_cds {
        let mut de_cds_mf = MatrixFull::new([3, natm], 0.0);
        for dir in 0..3 {
            for i in 0..natm {
                de_cds_mf[[dir, i]] = d_cds[dir][[i, 0]];
            }
        }
        println!("  grad_cds     done  (min={:12.6e}, max={:12.6e}, rms={:12.6e})",
            grad_stats(&de_cds_mf).0, grad_stats(&de_cds_mf).1, grad_stats(&de_cds_mf).2);
        if print_lvl >= 2 {
            println!("{}", format_grad_component("de_solvent.cds", &de_cds_mf, elem));
        }
        Some(de_cds_mf)
    } else {
        None
    };

    let mut de = de_nuc;
    for i in 0..de.data.len() { de.data[i] += de_qv.data[i] + de_solver.data[i]; }
    if let Some(ref de_cds_mf) = de_cds {
        for i in 0..de.data.len() { de.data[i] += de_cds_mf.data[i]; }
    }

    if print_lvl >= 2 {
        println!("{}", format_grad_component("de_solvent.total", &de, elem));
    }
    println!("PCM gradient total: {:.2} sec", t0.elapsed().as_secs_f64());
    de
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 grids, 2 atoms: verify grid/atom response cancellation for identity dM.
    #[test]
    fn test_antisym_chain_rule_identity() {
        let ngrids = 3;
        let natm = 2;
        let gslice: Vec<(usize, usize)> = vec![(0, 1), (1, 3)];

        let u = vec![1.0, 2.0, 3.0];
        let w = vec![4.0, 5.0, 6.0];

        // identity dM → dm_w = w, dmt_u = u → grid & atom cancel exactly
        let mut dm0 = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
        let mut dm1 = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
        let mut dm2 = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
        for i in 0..ngrids {
            dm0[(i, i)] = 1.0;
            dm1[(i, i)] = 1.0;
            dm2[(i, i)] = 1.0;
        }
        let dm = vec![dm0, dm1, dm2];

        let de = antisym_chain_rule(&u, &dm, &w, &gslice, natm);

        // Atom 0: u[0]*w[0] - w[0]*u[0] = 0
        // Atom 1: u[1]*w[1]+u[2]*w[2] - (w[1]*u[1]+w[2]*u[2]) = 0
        for xyz in 0..3 {
            assert!((de[(xyz, 0)]).abs() < 1e-12);
            assert!((de[(xyz, 1)]).abs() < 1e-12);
        }
    }

    /// 2×2 antisymmetric dM with hand-computed expected values.
    #[test]
    fn test_antisym_chain_rule_off_diag() {
        let ngrids = 2;
        let natm = 2;
        let gslice: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];

        let u = vec![1.0, 2.0];
        let w = vec![3.0, 4.0];

        // dM = [[0, a], [-a, 0]]
        let a_val = 2.0;
        let mut dm0 = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
        dm0[(0, 1)] = a_val;
        dm0[(1, 0)] = -a_val;
        let dm = vec![dm0.clone(), dm0.clone(), dm0.clone()];

        let de = antisym_chain_rule(&u, &dm, &w, &gslice, natm);

        // dM·w = [a*w1, -a*w0] = [8, -6]
        // dM^T·u = [-a*u1, a*u0] = [-4, 2]
        // Atom 0: grid=1*8=8, atom=3*(-4)=-12 → s=8-(-12)=20
        // Atom 1: grid=2*(-6)=-12, atom=4*2=8 → s=-12-8=-20
        for xyz in 0..3 {
            assert!((de[(xyz, 0)] - 20.0).abs() < 1e-12);
            assert!((de[(xyz, 1)] - (-20.0)).abs() < 1e-12);
        }
    }

    /// Verify diag_s_correction hand-computed values.
    #[test]
    fn test_diag_s_correction() {
        let ngrids = 2;
        let natm = 2;
        let u_q = vec![1.0, 2.0];
        let dsii_df = vec![0.5, 0.3];
        let mut df = MatrixFull::<f64>::new([ngrids, natm * 3], 0.0);
        df[(0, 0)] = 1.0;  // dF0/dR0^x = 1
        df[(1, 4)] = 1.0;  // dF1/dR1^y = 1

        let de = diag_s_correction(&u_q, &dsii_df, &df, natm);

        assert!((de[(0, 0)] - 0.5).abs() < 1e-12);  // 1*0.5*1
        assert!((de[(1, 1)] - 0.6).abs() < 1e-12);  // 2*0.3*1
        assert!((de[(1, 0)]).abs() < 1e-12);
        assert!((de[(0, 1)]).abs() < 1e-12);
    }

    /// Verify da_contract hand-computed values.
    #[test]
    fn test_da_contract() {
        let ngrids = 2;
        let natm = 2;
        let w = vec![2.0, 3.0];
        let mut da = MatrixFull::<f64>::new([ngrids, natm * 3], 0.0);
        da[(0, 0)] = 0.5;  // dA0/dR0^x
        da[(1, 3)] = 1.0;  // dA1/dR1^z

        let de = da_contract(&w, &da, natm);

        assert!((de[(0, 0)] - 1.0).abs() < 1e-12);  // 2*0.5
        assert!((de[(2, 1)] - 3.0).abs() < 1e-12);  // 3*1.0
    }
}
