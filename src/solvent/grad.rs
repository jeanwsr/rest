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
            let (ni, nj, nk) = (norm_vec[i].0, norm_vec[i].1, norm_vec[i].2);
            let (gix, giy, giz) = (grid_coords[i][0], grid_coords[i][1], grid_coords[i][2]);
            for j in 0..ngrids {
                if i == j { continue; }
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
                let nj_rij = ni*dx + nj*dy + nk*dz;
                let dD_dri = 4.0 * xr*xr * xij / PI.sqrt() * expv
                           * nj_rij / (r2 * r);
                for xyz in 0..3 {
                    let d = match xyz { 0 => dx, 1 => dy, _ => dz };
                    let dr = d / r;
                    let nc = match xyz { 0 => ni, 1 => nj, _ => nk };
                    dD_mat[xyz][(i, j)] = dD_dri * dr
                        + dSdr0 * (-nc/r + 3.0*nj_rij/r2 * dr);
                }
            }
        }
        Some(dD_mat)
    } else {
        None
    };

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
    let ngrids   = surface.surface_calc.grid_coords.len();
    let v_grids  = &pscf.v_grids;            // [ngrids]
    let K        = &pstatic.K;               // LU‑decomposed
    let K_ipiv   = &pstatic.K_ipiv;
    let q_sym    = &pscf.q_sym;              // [ngrids, 1]
    let f_eps    = pstatic.f_epsilon;

    // vK_1 = K^{-T} · v_grids   (solve transposed system)
    let vK_1 = solve_lu_transpose(K, K_ipiv, v_grids)
        .expect("grad_solvent_solver: solve_lu_transpose failed");

    let (dF, dA) = get_dF_dA(surface);

    let with_D = matches!(method, PcmMethod::IEFPCM | PcmMethod::SSVPE);
    let (_dD_opt, dS, dSii_dF) = get_dD_dS(surface, &dF, with_D);

    let mut de = MatrixFull::<f64>::new([3, natm], 0.0);

    match method {
        PcmMethod::CPCM | PcmMethod::COSMO => {
            // dK = dS,  dR = 0
            // dE_solver = ½ vK_1^T · dS · q_sym (anti‑symmetrised)
            //            + ½ Σ_i vK_1[i]·q_sym[i]·∂S_ii/∂R_a  (diagonal correction)
            for xyz in 0..3 {
                // BLAS: dS_q = dS[xyz] * q_sym  →  (dS_q)[i] = Σ_j dS[i][j] * q_sym[j]
                let dS_q = _dgemm_scaled(&dS[xyz], 'N', q_sym, 'N', 1.0);
                // BLAS: dST_vK1 = dS[xyz]^T * vK_1  →  (dST_vK1)[j] = Σ_i dS[i][j] * vK_1[i]
                let dST_vK1 = _dgemm_scaled(&dS[xyz], 'T', &vK_1, 'N', 1.0);

                for a in 0..natm {
                    let (p0, p1) = gslice[a];

                    // grid response – Σ_{i∈A} vK_1[i] · (dS · q_sym)[i]
                    let mut off = 0.0;
                    for i in p0..p1 { off += vK_1[i] * dS_q[(i, 0)]; }

                    // atom response – Σ_{j∈A} q_sym[j] · (dS^T · vK_1)[j]
                    for j in p0..p1 { off -= q_sym[(j, 0)] * dST_vK1[(j, 0)]; }

                    off *= 0.5;

                    // diagonal correction – Σ_i vK_1[i]·q_sym[i]·∂S_ii/∂R_a
                    // ∂S_ii/∂R_a = dSii_dF[i] · dF[(i, a*3+xyz)]
                    let mut diag = 0.0;
                    for i in 0..ngrids {
                        diag += vK_1[i] * q_sym[(i, 0)] * dSii_dF[i] * dF[(i, a*3 + xyz)];
                    }
                    diag *= 0.5;

                    de[(xyz, a)] = -(off + diag);
                }
            }
        }
        PcmMethod::IEFPCM => {
            // TODO: full D/dA terms
            let _fac = f_eps / (2.0 * PI);
            for xyz in 0..3 {
                let dS_q = _dgemm_scaled(&dS[xyz], 'N', q_sym, 'N', 1.0);
                let dST_vK1 = _dgemm_scaled(&dS[xyz], 'T', &vK_1, 'N', 1.0);
                for a in 0..natm {
                    let (p0, p1) = gslice[a];
                    let mut off = 0.0;
                    for i in p0..p1 { off += vK_1[i] * dS_q[(i, 0)]; }
                    for j in p0..p1 { off -= q_sym[(j, 0)] * dST_vK1[(j, 0)]; }
                    off *= 0.5;
                    let mut diag = 0.0;
                    for i in 0..ngrids {
                        diag += vK_1[i] * q_sym[(i, 0)] * dSii_dF[i] * dF[(i, a*3 + xyz)];
                    }
                    diag *= 0.5;
                    de[(xyz, a)] -= off + diag;
                }
            }
            eprintln!("IEFPCM gradient: dS contribution only — missing D/dA terms");
        }
        PcmMethod::SSVPE => {
            // TODO: full D/dA terms
            let _fac = f_eps / (4.0 * PI);
            for xyz in 0..3 {
                let dS_q = _dgemm_scaled(&dS[xyz], 'N', q_sym, 'N', 1.0);
                let dST_vK1 = _dgemm_scaled(&dS[xyz], 'T', &vK_1, 'N', 1.0);
                for a in 0..natm {
                    let (p0, p1) = gslice[a];
                    let mut off = 0.0;
                    for i in p0..p1 { off += vK_1[i] * dS_q[(i, 0)]; }
                    for j in p0..p1 { off -= q_sym[(j, 0)] * dST_vK1[(j, 0)]; }
                    off *= 0.5;
                    let mut diag = 0.0;
                    for i in 0..ngrids {
                        diag += vK_1[i] * q_sym[(i, 0)] * dSii_dF[i] * dF[(i, a*3 + xyz)];
                    }
                    diag *= 0.5;
                    de[(xyz, a)] -= off + diag;
                }
            }
            eprintln!("SSVPE gradient: dS contribution only — missing D/dA terms");
        }
    }

    de
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

    let mut de = de_nuc;
    for i in 0..de.data.len() { de.data[i] += de_qv.data[i] + de_solver.data[i]; }

    if print_lvl >= 2 {
        println!("{}", format_grad_component("de_solvent.total", &de, elem));
    }
    println!("PCM gradient total: {:.2} sec", t0.elapsed().as_secs_f64());
    de
}
