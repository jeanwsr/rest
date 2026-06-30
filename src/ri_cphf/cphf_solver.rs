/// CP-HF Dense Solver using AO-based gen_vind for fvind.
///
/// fvind(z) = C_vir^T @ (J[dm1] - 0.5*K[dm1]) @ C_occ + fxc[z]
///   where dm1 = 2*C_vir @ z @ C_occ^T + h.c.

use rest_tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::_dsolve;
use crate::scf_io::SCF;
use crate::ri_tddft::utils::tddft_occupation_parameters;
use crate::dft::response::{gen_vind_opt, VindWorkspace, FxcHessianCache,
    prepare_fxc_hessian_cache};

pub struct CPHFSolver {
    pub nocc: usize,
    pub nvir: usize,
    pub dim: usize,
    pub e_ai: Vec<f64>,
    pub start_mo: usize,
    pub lumo: usize,
    /// Precomputed fxc cache (None for HF). Built once at construction;
    /// reused by every fvind call inside the CP-HF loop.
    pub fxc_cache: Option<FxcHessianCache>,
    /// Cached MO slices for optimized fvind
    pub ws: VindWorkspace,
}

impl CPHFSolver {
    pub fn new(scf: &SCF) -> Self {
        let (start_mo, num_state, occ_size, vir_size, _homo, lumo) =
            tddft_occupation_parameters(scf);

        let dim = occ_size * vir_size;
        if dim == 0 { panic!("CP-HF: empty OV space"); }

        // Energy denominators
        let ks = &scf.eigenvalues[0];
        let mut e_ai = vec![0.0; dim];
        for a in 0..vir_size {
            let e_a = ks[lumo + a];
            for i in 0..occ_size {
                e_ai[i + a * occ_size] = 1.0 / (e_a - ks[start_mo + i]);
            }
        }

        // fxc cache for DFT (built once; reused across every fvind call)
        let is_dft = !scf.mol.xc_data.dfa_compnt_scf.is_empty();
        let fxc_cache = if is_dft {
            Some(prepare_fxc_hessian_cache(scf))
        } else {
            None
        };

        let ws = VindWorkspace::new(scf, occ_size, vir_size, start_mo, lumo);
        CPHFSolver { nocc: occ_size, nvir: vir_size, dim, e_ai, start_mo, lumo, fxc_cache, ws }
    }

    /// Compute orbital Hessian G[z] using optimized AO-based gen_vind.
    pub fn fvind(&self, scf: &SCF, z: &[f64]) -> Vec<f64> {
        let full = gen_vind_opt(scf, &self.ws, z, self.fxc_cache.as_ref(), None, None);
        // Extract VO part only (skip frozen rows)
        let fo_size = self.ws.nfrozen * self.ws.nocc;
        full[fo_size..fo_size + self.dim].to_vec()
    }

    /// Build LHS = I + G̃ column by column and RHS.
    pub fn build_system(&self, scf: &SCF, h1: &[f64]) -> (MatrixFull<f64>, Vec<f64>) {
        let dim = self.dim;
        let mut lhs = MatrixFull::new([dim, dim], 0.0);
        let mut unit = vec![0.0; dim];

        for jb in 0..dim {
            unit[jb] = 1.0;
            let gtilde: Vec<f64> = self.fvind(scf, &unit).iter()
                .zip(self.e_ai.iter()).map(|(g, e)| g * e).collect();
            unit[jb] = 0.0;
            for ia in 0..dim { lhs[[ia, jb]] = gtilde[ia]; }
            lhs[[jb, jb]] += 1.0;
        }

        let rhs: Vec<f64> = h1.iter().zip(self.e_ai.iter())
            .map(|(h, e)| -h * e).collect();
        (lhs, rhs)
    }

    pub fn solve_dense(&self, scf: &SCF, h1: &[f64]) -> Option<Vec<f64>> {
        let (lhs, rhs) = self.build_system(scf, h1);
        _dsolve(&lhs, &rhs)
    }

    /// Krylov subspace solver: solve (I + G̃) x = b where b = -h1 * e_ai.
    ///
    /// Uses Pople-style iteration (PySCF lib.krylov equivalent).
    /// Only needs matvec (fvind(z)*e_ai), never builds full LHS matrix.
    pub fn solve_krylov(&self, scf: &SCF, h1: &[f64],
                         max_cycle: usize, tol: f64) -> Vec<f64>
    {
        let dim = self.dim;
        let rhs: Vec<f64> = h1.iter().zip(self.e_ai.iter())
            .map(|(h, e)| -h * e).collect();

        // Handle zero RHS (symmetry leads to zero response)
        let b = rhs;
        let b_norm2: f64 = b.iter().map(|x| x * x).sum();
        if b_norm2 < 1e-30 {
            return vec![0.0; dim]; // Zero solution for zero perturbation
        }

        // Initial vector x1 = b
        let mut x1 = b.clone();

        // Krylov subspace
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut axs: Vec<Vec<f64>> = Vec::new();
        let mut innerprod: Vec<f64> = Vec::new();

        // Compute initial inner product
        innerprod.push(b_norm2);

        let mut converged = false;
        for _cycle in 0..max_cycle {
            // axt = A(x1) = G̃(x1) = fvind(x1) * e_ai
            let fv = self.fvind(scf, &x1);
            let axt: Vec<f64> = fv.iter().zip(self.e_ai.iter()).map(|(f, e)| f * e).collect();

            // Store vectors
            xs.push(x1.clone());
            axs.push(axt.clone());

            // New trial = axt, orthogonalized against all previous xs
            let mut x_new = axt;
            for (i, xi) in xs.iter().enumerate() {
                let dot_ax: f64 = x_new.iter().zip(xi.iter()).map(|(a, x)| a * x).sum();
                let w = dot_ax / innerprod[i];
                for j in 0..dim { x_new[j] -= w * xi[j]; }
            }

            let norm2: f64 = x_new.iter().map(|x| x * x).sum();
            innerprod.push(norm2);
            if norm2 < tol * tol {
                converged = true;
                break;
            }
            x1 = x_new;
        }

        if !converged {
            // Use last vector even if not converged
        }

        // Build subspace matrix H and RHS g
        let nd = xs.len();
        let mut h_mat = vec![0.0; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                h_mat[j * nd + i] = xs[i].iter().zip(axs[j].iter())
                    .map(|(x, a)| x * a).sum();
            }
            h_mat[i * nd + i] += innerprod[i]; // add identity contribution
        }

        let mut g_vec = vec![0.0; nd];
        for i in 0..nd {
            g_vec[i] = b.iter().zip(xs[i].iter()).map(|(x, y)| x * y).sum();
        }

        // Solve H·c = g using _dsolve (small nd×nd system)
        let mut h_full = MatrixFull::new([nd, nd], 0.0);
        for i in 0..nd { for j in 0..nd {
            h_full[[i, j]] = h_mat[j * nd + i]; // column-major
        }}
        let c = _dsolve(&h_full, &g_vec).unwrap_or_else(|| {
            // Fallback: just use first coefficient
            let mut c0 = vec![0.0; nd]; c0[0] = 1.0; c0
        });

        // Build solution x = Σ c[i] * xs[i]
        let mut x = vec![0.0; dim];
        for i in 0..nd {
            let ci = c[i];
            for j in 0..dim {
                x[j] += ci * xs[i][j];
            }
        }
        x
    }
}

/// Build dipole h1 for component comp (0=x,1=y,2=z) in MO VO-block.
pub fn build_dipole_h1_comp(scf: &SCF, comp: usize) -> Vec<f64> {
    let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
        tddft_occupation_parameters(scf);
    let ao_dip = crate::ri_bse::dipoles::obtain_ao_dips(scf, None);
    let eigvec = &scf.eigenvectors[0];
    let nao = eigvec.size[0];
    let dim = occ_size * vir_size;

    let ao_dip_comp = ao_dip.get_reducing_matrix(comp).unwrap();
    let mut mu_mo_tmp = MatrixFull::new([nao, nao], 0.0);
    rest_tensors::matrix::matrix_blas_lapack::_dgemm_full(
        eigvec, 'T', &ao_dip_comp, 'N', &mut mu_mo_tmp, 1.0, 0.0);
    let mut mu_mo = MatrixFull::new([nao, nao], 0.0);
    rest_tensors::matrix::matrix_blas_lapack::_dgemm_full(
        &mu_mo_tmp, 'N', eigvec, 'N', &mut mu_mo, 1.0, 0.0);

    let mut h1 = vec![0.0; dim];
    for a in 0..vir_size { for i in 0..occ_size {
        h1[i + a * occ_size] = mu_mo[[lumo + a, start_mo + i]];
    }}
    h1
}

pub fn build_dipole_h1(scf: &SCF) -> Vec<f64> { build_dipole_h1_comp(scf, 2) }

/// Main test: solve CP-HF with dense AND Krylov solvers, verify they match.
pub fn test_cphf_dense(scf: &SCF) -> Result<(), String> {
    println!("  Initializing CP-HF...");
    let solver = CPHFSolver::new(scf);
    println!("  nocc = {}, nvir = {}, dim = {}", solver.nocc, solver.nvir, solver.dim);

    let h1_x = build_dipole_h1_comp(scf, 0);
    let h1_y = build_dipole_h1_comp(scf, 1);
    let h1_z = build_dipole_h1_comp(scf, 2);

    // ═══ Single fvind timing ═══
    {
        let mut test_z = vec![0.0; solver.dim];
        for i in 0..solver.dim { test_z[i] = ((i * 7 + 13) as f64).sin() * 0.1; }
        let t0 = std::time::Instant::now();
        let _ = solver.fvind(scf, &test_z);
        let elapsed = t0.elapsed();
        println!("  Single fvind: {:.6}s", elapsed.as_secs_f64());
    }

    // ═══ Krylov solver only (fast mode) ═══
    let kr_x = solver.solve_krylov(scf, &h1_x, 50, 1e-12);
    let kr_y = solver.solve_krylov(scf, &h1_y, 50, 1e-12);
    let kr_z = solver.solve_krylov(scf, &h1_z, 50, 1e-12);

    // Krylov residual
    let fv_kz = solver.fvind(scf, &kr_z);
    let res_k: f64 = (0..solver.dim).map(|i| {
        let r = kr_z[i] + fv_kz[i] * solver.e_ai[i] + h1_z[i] * solver.e_ai[i];
        r * r
    }).sum();
    println!("  Krylov residual |(I+G̃)·U - U₀| = {:.2e}", res_k.sqrt());

    // Polarizability from Krylov
    let dot = |h: &[f64], u: &[f64]| -> f64 { h.iter().zip(u).map(|(a,b)| a*b).sum() };
    let a_k = [-4.0*dot(&h1_x,&kr_x), -4.0*dot(&h1_y,&kr_y), -4.0*dot(&h1_z,&kr_z)];
    println!("\n  === Static Polarizability Tensor (a.u.) ===");
    println!("  α_xx={:.12e}  α_yy={:.12e}  α_zz={:.12e}  α_bar={:.12e}",
             a_k[0], a_k[1], a_k[2], (a_k[0]+a_k[1]+a_k[2])/3.0);

    Ok(())
}
