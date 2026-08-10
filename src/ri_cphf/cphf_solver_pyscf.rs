/// PySCF-style CP-HF solver using (nmo, nocc) space.
///
/// Differences from CPHFSolver:
///   - Operates in full (nmo, nocc) space, not only (nvir, nocc)
///   - Supports s1 (overlap derivative) for field-dependent basis
///   - Wraps gen_vind_opt from dft/response.rs with zero-padding
///   - Batch RHS for multiple atom-displacement perturbations
///
/// Matches PySCF hessian/rhf.py gen_vind + cphf.solve convention.

use rest_tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dsolve};
use crate::scf_io::SCF;
use crate::ri_tddft::utils::tddft_occupation_parameters;
use crate::dft::response::{gen_vind_opt, gen_vind_opt_batched, VindWorkspace, FxcHessianCache, KLowRankPrecompute};

/// Solves (I + G̃)U = -(h1 - s1·e_i)·e_ai  in (nmo, nocc) space.
///
/// Column-major flat format for (nmo, nocc) vectors:
///   flat[row + col * nmo]    for MO row `row`, occupied column `col`
pub struct CPHFSolverPySCF {
    pub nmo: usize,           // total number of MOs
    pub nocc: usize,          // active occupied MOs
    pub nvir: usize,          // active virtual MOs
    pub nao: usize,           // number of AOs
    pub dim: usize,           // nocc * nvir (VO subspace dim)
    pub start_mo: usize,      // first active occupied MO index
    pub lumo: usize,          // first virtual MO index

    /// Full MO coefficient matrix [nao, nmo], column-major
    pub c_mo: MatrixFull<f64>,

    /// Workspace with C_occ, C_vir slices for gen_vind_opt
    pub ws: VindWorkspace,

    /// Energy denominators e_ai[a + i*nvir] = 1/(e_vir[a] - e_occ[i])
    /// flat index: i + a*nocc (same as CPHFSolver)
    pub e_ai: Vec<f64>,

    /// MO eigenvalues [nmo]
    pub mo_energy: Vec<f64>,

    /// MO occupation numbers [nmo]
    pub mo_occ: Vec<f64>,
}

impl CPHFSolverPySCF {
    pub fn new(scf: &SCF) -> Self {
        let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
            tddft_occupation_parameters(scf);

        let dim = occ_size * vir_size;
        if dim == 0 { panic!("CP-HF: empty OV space"); }

        let mo_energy = scf.eigenvalues[0].clone();
        let mo_occ = scf.occupation[0].clone();
        let nmo = mo_energy.len();
        let nao = scf.mol.num_basis;

        // Energy denominators
        let mut e_ai = vec![0.0; dim];
        for a in 0..vir_size {
            let e_a = mo_energy[lumo + a];
            for i in 0..occ_size {
                e_ai[i + a * occ_size] = 1.0 / (e_a - mo_energy[start_mo + i]);
            }
        }

        let ws = VindWorkspace::new(scf, occ_size, vir_size, start_mo, lumo);
        let c_mo = scf.eigenvectors[0].clone();

        CPHFSolverPySCF {
            nmo, nocc: occ_size, nvir: vir_size, nao, dim,
            start_mo, lumo,
            c_mo, ws, e_ai,
            mo_energy, mo_occ,
        }
    }

    /// Create a CPHFSolverPySCF with full occupation (no frozen core).
    /// Matches PySCF's kernel() behavior which uses all occupied orbitals.
    pub fn new_full(scf: &SCF) -> Self {
        let mo_energy = scf.eigenvalues[0].clone();
        let mo_occ = scf.occupation[0].clone();
        let nmo = mo_energy.len();
        let nao = scf.mol.num_basis;
        let homo = scf.homo[0] as usize;
        let lumo = scf.lumo[0] as usize;
        let start_mo = 0; // include all occupied orbitals
        let occ_size = homo + 1; // full occupation count
        let vir_size = nmo - lumo;
        let dim = occ_size * vir_size;

        // Energy denominators
        let mut e_ai = vec![0.0; dim];
        for a in 0..vir_size {
            let e_a = mo_energy[lumo + a];
            for i in 0..occ_size {
                e_ai[i + a * occ_size] = 1.0 / (e_a - mo_energy[start_mo + i]);
            }
        }

        let ws = VindWorkspace::new(scf, occ_size, vir_size, start_mo, lumo);
        let c_mo = scf.eigenvectors[0].clone();

        CPHFSolverPySCF {
            nmo, nocc: occ_size, nvir: vir_size, nao, dim,
            start_mo, lumo,
            c_mo, ws, e_ai,
            mo_energy, mo_occ,
        }
    }

    // ── Index conversions ──────────────────────────────────────────────

    /// Convert flat (nmo, nocc) index to VO flat index.
    /// Column-major: (nmo,nocc) flat = row + col * nmo
    /// VO flat = i + a * nocc
    fn vo_index(&self, row: usize, col: usize) -> Option<usize> {
        let a = row.checked_sub(self.lumo)?;
        if a >= self.nvir { return None; }
        let i = col.checked_sub(self.start_mo)?;
        if i >= self.nocc { return None; }
        Some(i + a * self.nocc)
    }

    /// Check if (row, col) is in the VO block of (nmo, nocc) space.
    fn is_vo(&self, row: usize, col: usize) -> bool {
        row >= self.lumo && row < self.lumo + self.nvir
            && col >= self.start_mo && col < self.start_mo + self.nocc
    }

    /// Check if (row, col) is in the occ-occ block.
    fn is_oo(&self, row: usize, col: usize) -> bool {
        row >= self.start_mo && row < self.start_mo + self.nocc
            && col >= self.start_mo && col < self.start_mo + self.nocc
    }

    // ── fvind wrapper for (nmo, nocc) space ────────────────────────────

    /// Compute fvind in (nmo, nocc) space.
    ///
    /// `z_full`: flat vector of length `nmo * nocc`, column-major.
    ///   element (row, col) = z_full[row + col * nmo]
    ///
    /// Returns response of same shape. Only the VO block is non-zero
    /// (computed by gen_vind_opt); all other elements are zero.
    pub fn fvind_nmo_nocc(&self, scf: &SCF, fxc_cache: Option<&FxcHessianCache>, z_full: &[f64]) -> Vec<f64> {
        // Extract VO block from (nmo, nocc) input
        let mut z_vo = vec![0.0; self.dim];
        for col in 0..self.nocc {
            for a in 0..self.nvir {
                let row = self.lumo + a;
                let nmc_idx = row + col * self.nmo;
                z_vo[col + a * self.nocc] = z_full[nmc_idx];
            }
        }

        // Extract OO block
        let mut z_oo = vec![0.0; self.nocc * self.nocc];
        for j in 0..self.nocc { for i in 0..self.nocc {
            let nmc_idx = (self.start_mo + i) + j * self.nmo;
            z_oo[i + j * self.nocc] = z_full[nmc_idx];
        }}

        // Extract frozen-occ block
        let nfrozen = self.ws.nfrozen;
        let mut z_fo = vec![0.0; nfrozen * self.nocc];
        for j in 0..self.nocc { for k in 0..nfrozen {
            // frozen MO row k, occ column j
            let nmc_idx = k + j * self.nmo;
            z_fo[j + k * self.nocc] = z_full[nmc_idx];
        }}

        // Compute full response: [frozen_resp (nfrozen*nocc), VO_resp (nvir*nocc)]
        let full_resp = gen_vind_opt(scf, &self.ws, &z_vo, fxc_cache, Some(&z_oo), Some(&z_fo));

        // Pad to full (nmo, nocc): first nfrozen*nocc = frozen response, then VO response
        let fo_size = nfrozen * self.nocc;
        let mut result = vec![0.0; self.nmo * self.nocc];
        // Frozen row response: rows 0..nfrozen
        for col in 0..self.nocc { for k in 0..nfrozen {
            result[k + col * self.nmo] = full_resp[col + k * self.nocc];
        }}
        // VO row response: rows lumo..lumo+nvir
        for col in 0..self.nocc {
            for a in 0..self.nvir {
                let row = self.lumo + a;
                let nmc_idx = row + col * self.nmo;
                result[nmc_idx] = full_resp[fo_size + col + a * self.nocc];
            }
        }
        result
    }

    // ── RHS builder with s1 ────────────────────────────────────────────

    /// Build VO-subspace RHS vector including s1 (overlap derivative).
    ///
    /// Equation: (I + G̃)U = b where b = -(h1 - s1·e_i)·e_ai
    /// Only the VO block matters; occ-occ block is handled separately.
    ///
    /// h1_nmc: h1 in (nmo, nocc) flat format
    /// s1_nmc: s1 in (nmo, nocc) flat format
    pub fn build_rhs_with_s1(&self, h1_nmc: &[f64], s1_nmc: &[f64]) -> Vec<f64> {
        let mut rhs = vec![0.0; self.dim];
        for col in 0..self.nocc {
            let occ_idx = self.start_mo + col;
            let e_occ = self.mo_energy[occ_idx];
            for a in 0..self.nvir {
                let row = self.lumo + a;
                let nmc_idx = row + col * self.nmo;
                let h1_val = h1_nmc[nmc_idx];
                let s1_val = s1_nmc[nmc_idx];
                let vo_idx = col + a * self.nocc;
                rhs[vo_idx] = -(h1_val - s1_val * e_occ) * self.e_ai[vo_idx];
            }
        }
        rhs
    }

    /// Compute occ-occ block of solution from s1: U_ij = -0.5 * s1_ij.
    ///
    /// s1_nmc: s1 in (nmo, nocc) flat format.
    /// Returns flat vector of length nocc * nocc, column-major: U[i + j*nocc]
    pub fn solve_occ_occ_from_s1(&self, s1_nmc: &[f64]) -> Vec<f64> {
        let mut u_oo = vec![0.0; self.nocc * self.nocc];
        for j in 0..self.nocc {
            for i in 0..self.nocc {
                let row = self.start_mo + i;
                let nmc_idx = row + j * self.nmo;
                u_oo[i + j * self.nocc] = -0.5 * s1_nmc[nmc_idx];
            }
        }
        u_oo
    }

    /// Assemble full (nmo, nocc) solution from VO solution and occ-occ block.
    pub fn assemble_full_solution(&self, u_vo: &[f64], u_oo: &[f64]) -> Vec<f64> {
        let mut u_full = vec![0.0; self.nmo * self.nocc];
        // Fill VO block
        for col in 0..self.nocc {
            for a in 0..self.nvir {
                let row = self.lumo + a;
                let nmc_idx = row + col * self.nmo;
                u_full[nmc_idx] = u_vo[col + a * self.nocc];
            }
        }
        // Fill occ-occ block (if s1 was non-zero)
        // occ-occ block: column index j (0..nocc), row = start_mo+i
        if u_oo.len() >= self.nocc * self.nocc {
            for j in 0..self.nocc {
                for i in 0..self.nocc {
                    let row = self.start_mo + i;
                    let nmc_idx = row + j * self.nmo;
                    u_full[nmc_idx] = u_oo[i + j * self.nocc];
                }
            }
        }
        u_full
    }

    /// Assemble full (nmo, nocc) solution from VO, OO, and frozen-occ parts.
    fn assemble_full_solution_with_frozen(&self, u_vo: &[f64], u_oo: &[f64], u_frozen: &[f64]) -> Vec<f64> {
        let mut u_full = vec![0.0; self.nmo * self.nocc];
        // Fill VO block
        for col in 0..self.nocc {
            for a in 0..self.nvir {
                let row = self.lumo + a;
                let nmc_idx = row + col * self.nmo;
                u_full[nmc_idx] = u_vo[col + a * self.nocc];
            }
        }
        // Fill occ-occ block
        if u_oo.len() >= self.nocc * self.nocc {
            for j in 0..self.nocc {
                for i in 0..self.nocc {
                    let row = self.start_mo + i;
                    let nmc_idx = row + j * self.nmo;
                    u_full[nmc_idx] = u_oo[i + j * self.nocc];
                }
            }
        }
        // Fill frozen-occ block (rows 0..start_mo, all occ columns)
        let nfrozen = self.start_mo;
        for col in 0..self.nocc {
            for k in 0..nfrozen {
                let nmc_idx = k + col * self.nmo;
                u_full[nmc_idx] = u_frozen[col + k * self.nocc];
            }
        }
        u_full
    }

    // ── Dense solver ───────────────────────────────────────────────────

    /// Dense solver: build augmented (I + G̃) matrix for (frozen + active VO) subspace.
    ///
    /// Includes frozen-occupied rows matching PySCF's solve_withs1 convention.
    /// Returns solution in (nmo, nocc) flat format.
    pub fn solve_dense(&self, scf: &SCF, fxc_cache: Option<&FxcHessianCache>,
                        h1_nmc: &[f64], s1_nmc: &[f64]) -> Option<Vec<f64>> {
        let dim = self.dim;
        let nocc = self.nocc;
        let nmo = self.nmo;
        let start_mo = self.start_mo;
        let nfrozen = start_mo;  // MOs 0..start_mo are frozen

        // Build RHS (VO only): b_vo[a,i] = -(h1 - s1*e_i) * e_ai + OO correction
        let mut rhs = self.build_rhs_with_s1(h1_nmc, s1_nmc);
        // Add OO correction: subtract fvind(mo1_oo) * e_ai from RHS (occ-occ block response)
        // where mo1_oo[i,j] = -0.5 * s1_nmc for active occupied rows
        {
            let mut z_oo_full = vec![0.0; nmo * nocc];
            for j in 0..nocc { for i in 0..nocc {
                let nmc_idx = (start_mo + i) + j * nmo;
                z_oo_full[nmc_idx] = -0.5 * s1_nmc[nmc_idx];
            }}
            let oo_resp_full = self.fvind_nmo_nocc(scf, fxc_cache, &z_oo_full);
            for ia in 0..dim {
                let ia_col = ia / nocc;
                let ia_row = ia % nocc;
                let r = self.lumo + ia_col;
                rhs[ia] -= oo_resp_full[r + ia_row * nmo] * self.e_ai[ia];
            }
        }

        // Build LHS: dim × dim VO-only system (I + G̃)
        let mut lhs = MatrixFull::new([dim, dim], 0.0);
        let mut z_full = vec![0.0; nmo * nocc];

        for jb in 0..dim {
            z_full.fill(0.0);
            let a_vir = jb / nocc;
            let occ_j = jb % nocc;
            let row = self.lumo + a_vir;
            z_full[row + occ_j * nmo] = 1.0;

            let resp_full = self.fvind_nmo_nocc(scf, fxc_cache, &z_full);

            for ia in 0..dim {
                let ia_col = ia / nocc;
                let ia_row = ia % nocc;
                let r = self.lumo + ia_col;
                lhs[[ia, jb]] = resp_full[r + ia_row * nmo] * self.e_ai[ia];
            }
            lhs[[jb, jb]] += 1.0; // identity
        }

        // Solve VO system
        let u_vo = _dsolve(&lhs, &rhs)?;

        // u_frozen from s1 (frozen orbitals don't rotate independently):
        // u_frozen[k, i] = -0.5 * s1[k, i] for frozen MO k, active occupied i
        let u_frozen: Vec<f64> = {
            let mut uf = vec![0.0; nfrozen * nocc];
            for k in 0..nfrozen {
                for i in 0..nocc {
                    let nmc_idx = k + i * nmo;
                    uf[i + k * nocc] = -0.5 * s1_nmc[nmc_idx];
                }
            }
            uf
        };

        // Assemble full (nmo, nocc) solution
        let u_oo = self.solve_occ_occ_from_s1(s1_nmc);
        Some(self.assemble_full_solution_with_frozen(&u_vo, &u_oo, &u_frozen))
    }

    // ── Krylov solver (Pople-style) ────────────────────────────────────

    /// Krylov subspace solver in (nmo, nocc) space.
    ///
    /// Iterates over the VO subspace only. Occ-occ block filled from s1.
    /// Returns solution in (nmo, nocc) flat format.
    pub fn solve_krylov(&self, scf: &SCF, fxc_cache: Option<&FxcHessianCache>,
                         h1_nmc: &[f64], s1_nmc: &[f64],
                         max_cycle: usize, tol: f64) -> Vec<f64> {
        let dim = self.dim;
        let debug = std::env::var("REST_CPHF_KRYLOV_DEBUG").is_ok()
            || scf.mol.ctrl.print_level >= 1;

        // Build RHS
        let mut b = self.build_rhs_with_s1(h1_nmc, s1_nmc);

        // OO correction: subtract fvind(mo1_oo) * e_ai from RHS.
        // The occ-occ block is fixed at mo1_oo[i,j] = -0.5 * s1[i,j]. Since the
        // Krylov matvec only operates on the VO block (z_full OO = 0), the
        // response of the OO block must be moved to the RHS, matching
        // solve_dense (cphf_solver_pyscf.rs lines 357-379) and PySCF's
        // solve_withs1 convention.
        {
            let mut z_oo_full = vec![0.0; self.nmo * self.nocc];
            for j in 0..self.nocc {
                for i in 0..self.nocc {
                    let nmc_idx = (self.start_mo + i) + j * self.nmo;
                    z_oo_full[nmc_idx] = -0.5 * s1_nmc[nmc_idx];
                }
            }
            let oo_resp_full = self.fvind_nmo_nocc(scf, fxc_cache, &z_oo_full);
            for ia in 0..dim {
                let ia_col = ia / self.nocc;  // virtual index
                let ia_row = ia % self.nocc;  // occupied index
                let r = self.lumo + ia_col;
                let g_oo = oo_resp_full[r + ia_row * self.nmo];
                b[ia] -= g_oo * self.e_ai[ia];
            }
        }

        let b_norm2: f64 = b.iter().map(|x| x * x).sum();
        if b_norm2 < 1e-30 {
            return self.assemble_full_solution(
                &vec![0.0; dim],
                &self.solve_occ_occ_from_s1(s1_nmc));
        }

        // Matvec for VO subspace: z_vo → G̃(z_vo)
        let matvec = |z_vo: &[f64]| -> Vec<f64> {
            let mut z_full = vec![0.0; self.nmo * self.nocc];
            for i in 0..dim {
                let a = i / self.nocc;     // virtual index
                let occ = i % self.nocc;    // occupied index
                let r = self.lumo + a;
                z_full[r + occ * self.nmo] = z_vo[i];
            }
            let resp_full = self.fvind_nmo_nocc(scf, fxc_cache, &z_full);
            let mut g_vo = vec![0.0; dim];
            for i in 0..dim {
                let col = i / self.nocc;
                let row = i % self.nocc;
                let r = self.lumo + col;
                let c = row; // 0-indexed column
                g_vo[i] = resp_full[r + c * self.nmo] * self.e_ai[i];
            }
            g_vo
        };

        // Initial vector
        let mut x1 = b.clone();

        // Krylov subspace
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut axs: Vec<Vec<f64>> = Vec::new();
        let mut innerprod: Vec<f64> = Vec::new();
        innerprod.push(b_norm2);

        for cycle in 0..max_cycle {
            let axt = matvec(&x1);
            xs.push(x1.clone());
            axs.push(axt.clone());

            // Orthogonalize new trial against all previous xs (CGS)
            let mut x_new = axt;
            for (i, xi) in xs.iter().enumerate() {
                let dot_ax: f64 = x_new.iter().zip(xi.iter()).map(|(a, x)| a * x).sum();
                let w = dot_ax / innerprod[i];
                for j in 0..dim { x_new[j] -= w * xi[j]; }
            }

            let norm2: f64 = x_new.iter().map(|x| x * x).sum();
            innerprod.push(norm2);
            if debug {
                println!("    CP-HF iteration {}: residual={:.4e}", cycle, norm2.sqrt());
            }
            if norm2 < tol * tol { break; }
            x1 = x_new;
        }

        // Build subspace matrix H and RHS g
        let nd = xs.len();
        if debug {
            println!("    krylov: dim={}, nd={}, b_norm={:.4e}", dim, nd, b_norm2.sqrt());
        }
        let mut h_mat = vec![0.0; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                h_mat[j * nd + i] = xs[i].iter().zip(axs[j].iter())
                    .map(|(x, a)| x * a).sum();
            }
            h_mat[i * nd + i] += innerprod[i]; // I + G̃
        }

        let mut g_vec = vec![0.0; nd];
        for i in 0..nd {
            g_vec[i] = b.iter().zip(xs[i].iter()).map(|(x, y)| x * y).sum();
        }

        // Solve H·c = g
        let mut h_full = MatrixFull::new([nd, nd], 0.0);
        for i in 0..nd { for j in 0..nd {
            h_full[[i, j]] = h_mat[j * nd + i];
        }}
        let c = _dsolve(&h_full, &g_vec).unwrap_or_else(|| {
            let mut c0 = vec![0.0; nd]; c0[0] = 1.0; c0
        });

        // Build VO solution x = Σ c[i] * xs[i]
        let mut u_vo = vec![0.0; dim];
        for i in 0..nd {
            let ci = c[i];
            for j in 0..dim { u_vo[j] += ci * xs[i][j]; }
        }

        // Diagnostic: compute residual ||(I + G̃)x - b||
        if debug {
            let gx = matvec(&u_vo);
            let mut res2 = 0.0;
            for i in 0..dim {
                let r = u_vo[i] + gx[i] - b[i];
                res2 += r * r;
            }
            println!("    krylov residual ||(I+G)x - b|| = {:.4e}", res2.sqrt());
        }

        // Assemble full solution
        let u_oo = self.solve_occ_occ_from_s1(s1_nmc);
        self.assemble_full_solution(&u_vo, &u_oo)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Phase A: Batched interleaved Krylov solver for the CP-HF Hessian.
    //
    // Processes all 3*natom RHS vectors simultaneously, with each Krylov
    // cycle calling the matvec ONCE for all active RHS via
    // `gen_vind_opt_batched` (which batches the fxc response across RHS).
    // This mirrors PySCF's pattern of stacking RHS into one solve call.
    //
    // Each RHS maintains its own Krylov subspace and converges independently.
    // Converged RHS are removed from the active set so we don't waste matvec
    // work on them.
    // ─────────────────────────────────────────────────────────────────────

    /// Build RHS for the VO-subspace Krylov system (I + G̃)U = b, including
    /// the OO-block correction. The OO correction accounts for the response
    /// of the (fixed) occ-occ block mo1_oo = -0.5·s1 to the perturbation;
    /// this is moved to the RHS because the Krylov operates on VO only.
    ///
    /// Extracted from `solve_krylov` so that the batched solver can build
    /// all RHS upfront before the Krylov loop.
    pub fn build_rhs_with_oo_correction(
        &self, scf: &SCF, fxc_cache: Option<&FxcHessianCache>,
        h1_nmc: &[f64], s1_nmc: &[f64],
    ) -> Vec<f64> {
        let dim = self.dim;
        let mut b = self.build_rhs_with_s1(h1_nmc, s1_nmc);
        let mut z_oo_full = vec![0.0; self.nmo * self.nocc];
        for j in 0..self.nocc {
            for i in 0..self.nocc {
                let nmc_idx = (self.start_mo + i) + j * self.nmo;
                z_oo_full[nmc_idx] = -0.5 * s1_nmc[nmc_idx];
            }
        }
        let oo_resp_full = self.fvind_nmo_nocc(scf, fxc_cache, &z_oo_full);
        for ia in 0..dim {
            let ia_col = ia / self.nocc;
            let ia_row = ia % self.nocc;
            let r = self.lumo + ia_col;
            let g_oo = oo_resp_full[r + ia_row * self.nmo];
            b[ia] -= g_oo * self.e_ai[ia];
        }
        b
    }

    /// Batched VO-subspace matvec. Returns G̃(z_vo) per RHS.
    /// z_vo_batch: n_rhs × dim (VO block only; OO/FO blocks are zero).
    fn matvec_vo_batched(
        &self,
        scf: &SCF,
        fxc_cache: Option<&FxcHessianCache>,
        z_vo_batch: &[&[f64]],
        k_lowrank: Option<&KLowRankPrecompute>,
    ) -> Vec<Vec<f64>> {
        let n_rhs = z_vo_batch.len();
        // Direct batched response: z_vo_batch is already in the (nvir*nocc)
        // layout expected by gen_vind_opt_batched. OO/FO are zero (None); the
        // low-rank K path is used when the precomputation is available.
        let resp_full_batch = gen_vind_opt_batched(
            scf, &self.ws, z_vo_batch, fxc_cache, None, None, k_lowrank,
        );

        // Each resp_full is laid out as [frozen_resp(fo_size), VO_resp(dim)].
        // VO_resp[i + a*nocc] = vo_result[i, a] where i=occ, a=vir.
        // z_vo index k = i + a*nocc (matching build_rhs_with_s1 indexing).
        let fo_size = self.ws.nfrozen * self.nocc;
        let mut g_vo_batch = Vec::with_capacity(n_rhs);
        for i in 0..n_rhs {
            let resp_full = &resp_full_batch[i];
            let mut g_vo = vec![0.0; self.dim];
            for k in 0..self.dim {
                // VO_resp entry for index k is at offset fo_size + k.
                g_vo[k] = resp_full[fo_size + k] * self.e_ai[k];
            }
            g_vo_batch.push(g_vo);
        }
        g_vo_batch
    }

    /// Batched block Krylov solver with **shared subspace** (PySCF style).
    ///
    /// Mirrors `pyscf.lib.linalg_helper.krylov`: ALL RHS share a single
    /// Krylov subspace. Each cycle adds `n_active` vectors to the shared
    /// basis, where `n_active` is determined by QR-reducing the residual
    /// vectors of all currently-active RHS. This means every RHS benefits
    /// from every other RHS's search directions — for atom-displacement
    /// perturbations that span a similar subspace, this converges in
    /// roughly the same number of cycles as a single-RHS solve.
    ///
    /// Algorithm:
    ///   1. Initial QR of all RHS → orthogonal basis vectors
    ///   2. Per cycle:
    ///        a. block matvec on active vectors
    ///        b. extend shared subspace
    ///        c. CGS against full shared history (using original axt)
    ///        d. QR + threshold → new active set
    ///   3. Single projected solve H·C = G (one H, n_rhs RHS columns)
    ///
    /// Returns one VO-block solution per RHS.
    pub fn solve_krylov_batched(
        &self,
        scf: &SCF,
        fxc_cache: Option<&FxcHessianCache>,
        rhs_all: &[Vec<f64>],
        max_cycle: usize,
        tol: f64,
    ) -> Vec<Vec<f64>> {
        let n_rhs = rhs_all.len();
        let dim = self.dim;
        let debug = std::env::var("REST_CPHF_KRYLOV_DEBUG").is_ok()
            || scf.mol.ctrl.print_level >= 1;
        let profile = std::env::var("REST_CPHF_PROFILE").is_ok() || debug;
        // Match PySCF krylov's lindep threshold (DSOLVE_LINDEP default = 1e-13).
        let lindep: f64 = 1e-13;
        let tol2 = tol * tol;

        // ══ Shared Krylov subspace (single basis for ALL RHS) ════════════
        let mut xs: Vec<Vec<f64>> = Vec::new();     // basis vectors
        let mut axs: Vec<Vec<f64>> = Vec::new();    // A·basis vectors
        let mut innerprod: Vec<f64> = Vec::new();   // ||xs[i]||²

        // ══ Initial QR: orthogonalize RHS into shared basis ══════════════
        // Matches PySCF: x1, rmat = _qr(rhs); x1 *= rmat.diagonal()[:,None];
        //                innerprod = rmat.diagonal()**2
        let (mut x1, init_innerprod) = krylov_qr(rhs_all, lindep);
        innerprod.extend(init_innerprod);

        // PySCF termination: if initial RHS are essentially zero, return zeros.
        let max_init = innerprod.iter().fold(0.0f64, |a, &b| a.max(b));
        if max_init < lindep || max_init < tol2 {
            return (0..n_rhs).map(|_| vec![0.0; dim]).collect();
        }

        let mut total_matvecs: usize = 0;
        let mut cycles_done: usize = 0;

        if std::env::var("REST_MEM_TRACE").is_ok() {
            eprintln!("MEMTRACE cphf-solve-start  RSS = {:.1} MiB", crate::hessian::memory_monitor::current_rss_mb());
        }
        // Low-rank exchange-response precomputation (ground-state, built once
        // per solve; None if no RI tensor or not applicable).
        let k_lowrank = KLowRankPrecompute::new(scf, &self.ws);
        if std::env::var("REST_MEM_TRACE").is_ok() {
            eprintln!("MEMTRACE cphf-lowrank-built RSS = {:.1} MiB", crate::hessian::memory_monitor::current_rss_mb());
        }

        for cycle in 0..max_cycle {
            if x1.is_empty() { break; }
            let n_active = x1.len();
            total_matvecs += n_active;
            cycles_done = cycle + 1;

            // ── Block matvec on active vectors ───────────────────────────
            let x1_refs: Vec<&[f64]> = x1.iter().map(|v| &v[..]).collect();
            let axt_batch = self.matvec_vo_batched(scf, fxc_cache, &x1_refs, k_lowrank.as_ref());

            // ── Extend shared subspace with current active vectors ───────
            for k in 0..n_active {
                xs.push(x1[k].clone());
                axs.push(axt_batch[k].clone());
            }

            // ── CGS against full shared history ──────────────────────────
            // PySCF uses `axt` (original matvec output) for projection
            // coefficients, making this classical GS (numerically adequate
            // because xs are mutually orthogonal with known ||xs[i]||²).
            let mut x_new: Vec<Vec<f64>> = axt_batch.clone();
            for k in 0..x_new.len() {
                for (i, xsi) in xs.iter().enumerate() {
                    let dot_ax: f64 = axt_batch[k].iter().zip(xsi.iter()).map(|(a, b)| a * b).sum();
                    let w = dot_ax / innerprod[i];
                    for j in 0..dim { x_new[k][j] -= w * xsi[j]; }
                }
            }

            // ── QR + threshold → new active set ──────────────────────────
            let (x_new_orth, innerprod_new) = krylov_qr(&x_new, lindep);
            let max_innerprod = innerprod_new.iter().fold(0.0f64, |a, &b| a.max(b));

            if debug || profile {
                println!("    CP-HF iteration {}: residual={:.4e} (n_active={}/{})",
                    cycle, max_innerprod.sqrt(), n_active, x_new_orth.len());
            }

            if max_innerprod < lindep || max_innerprod < tol2 {
                break;
            }

            // Keep only directions above threshold (PySCF mask).
            let mut kept: Vec<Vec<f64>> = Vec::with_capacity(x_new_orth.len());
            let mut kept_innerprod: Vec<f64> = Vec::with_capacity(x_new_orth.len());
            for i in 0..x_new_orth.len() {
                if innerprod_new[i] > lindep && innerprod_new[i] > tol2 {
                    kept.push(x_new_orth[i].clone());
                    kept_innerprod.push(innerprod_new[i]);
                }
            }
            x1 = kept;
            innerprod.extend(kept_innerprod);
        }

        if profile {
            println!("    krylov_batched: n_rhs={}, cycles={}, shared subspace size={}, total matvecs={}",
                n_rhs, cycles_done, xs.len(), total_matvecs);
        }

        // ══ Single projected solve: H·C = G (one H, multiple RHS) ═══════
        let nd = xs.len();
        let mut u_vo: Vec<Vec<f64>> = (0..n_rhs).map(|_| vec![0.0; dim]).collect();
        if nd == 0 { return u_vo; }

        // Build H[nd,nd]: H[i,j] = xs[i]·axs[j]; H[i,i] += innerprod[i] (+I).
        let mut h = MatrixFull::new([nd, nd], 0.0);
        for i in 0..nd {
            for j in 0..nd {
                let v: f64 = xs[i].iter().zip(axs[j].iter()).map(|(a, b)| a * b).sum();
                h[[i, j]] = v;
            }
            h[[i, i]] += innerprod[i];
        }

        // For each RHS: g[i] = rhs[k]·xs[i]; solve H·c = g; reconstruct.
        // Note: H is identical for all RHS; we trade 36 small factorizations
        // for one shared H build. The total cost is negligible vs matvec time.
        for k in 0..n_rhs {
            let mut g = vec![0.0; nd];
            for i in 0..nd {
                g[i] = xs[i].iter().zip(rhs_all[k].iter()).map(|(a, b)| a * b).sum();
            }
            let c = _dsolve(&h, &g).unwrap_or_else(|| {
                let mut c0 = vec![0.0; nd]; c0[0] = 1.0; c0
            });
            for i in 0..nd {
                let ci = c[i];
                for j in 0..dim { u_vo[k][j] += ci * xs[i][j]; }
            }
        }

        u_vo
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PySCF-style QR for Krylov basis orthogonalization.
// Mirrors `pyscf.lib.linalg_helper._qr` (MGS against existing orthonormal
// vectors). Returns scaled (non-normalized) orthogonal vectors and their
// squared norms, matching PySCF's `x1 *= rmat.diagonal()[:,None]`;
// `innerprod = rmat.diagonal()**2` convention.
// ═══════════════════════════════════════════════════════════════════════
fn krylov_qr(vecs: &[Vec<f64>], lindep: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let nvec = vecs.len();
    if nvec == 0 { return (vec![], vec![]); }
    let dim = vecs[0].len();

    let mut qs: Vec<Vec<f64>> = Vec::with_capacity(nvec);  // orthonormal unit vectors
    let mut norms: Vec<f64> = Vec::with_capacity(nvec);    // ||v_i|| before normalization

    for i in 0..nvec {
        let mut xi = vecs[i].clone();
        // MGS: xi is updated in-place by each projection (matches PySCF _qr).
        for j in 0..qs.len() {
            let prod: f64 = xi.iter().zip(qs[j].iter()).map(|(a, b)| a * b).sum();
            for k in 0..dim { xi[k] -= qs[j][k] * prod; }
        }
        let innerprod: f64 = xi.iter().map(|v| v * v).sum();
        if innerprod > lindep {
            let norm = innerprod.sqrt();
            for k in 0..dim { xi[k] /= norm; }
            qs.push(xi);
            norms.push(norm);
        }
        // else: linearly dependent direction, drop.
    }

    // Scale qs by norm → orthogonal-but-not-normalized vectors, matching
    // PySCF's `x1 *= rmat.diagonal()[:,None]`.
    let x1: Vec<Vec<f64>> = qs.iter().enumerate()
        .map(|(i, q)| q.iter().map(|v| v * norms[i]).collect())
        .collect();
    let innerprod_out: Vec<f64> = norms.iter().map(|d| d * d).collect();

    (x1, innerprod_out)
}

// ═══════════════════════════════════════════════════════════════════════
// Helper: transform h1ao from AO to MO (nmo, nocc) format
// ═══════════════════════════════════════════════════════════════════════

/// Transform h1ao (one atom, shape [3*nao, nao]) to 3 (nmo, nocc) vectors.
///
/// h1ao_mat: MatrixFull<f64> with shape [3*nao, nao], column-major.
///   The x-direction block is at rows x*nao..(x+1)*nao, all columns.
///
/// Returns: Vec of 3 vectors, each length nmo * nocc, column-major.
pub fn transform_h1ao_ao2mo(
    solver: &CPHFSolverPySCF,
    h1ao_mat: &MatrixFull<f64>,
) -> Vec<Vec<f64>> {
    let nao = solver.nao;
    let nmo = solver.nmo;
    let nocc = solver.nocc;
    let start_mo = solver.start_mo;
    let c_mo = &solver.c_mo;

    let mut results = Vec::with_capacity(3);

    for x in 0..3 {
        // Extract [nao, nao] block for direction x
        // h1ao_mat is [3*nao, nao] column-major
        let mut h1_block = MatrixFull::new([nao, nao], 0.0);
        for i in 0..nao {
            for j in 0..nao {
                h1_block[[i, j]] = h1ao_mat[[x * nao + i, j]];
            }
        }

        // tmp[nao, nmo] = h1_block^T @ c_mo
        // We need C^T @ h1_block, so compute c_mo^T @ h1_block
        let mut tmp = MatrixFull::new([nao, nmo], 0.0);
        _dgemm_full(c_mo, 'T', &h1_block, 'N', &mut tmp, 1.0, 0.0);
        // tmp[p, q] = Σ_μ C[μ,p] * h1_block[μ,q]

        // result[nmo, nocc] = tmp @ C_occ(:, start_mo..start_mo+nocc)
        // Extract occupied columns from c_mo
        // c_occ_slice shape [nao, nocc]
        let mut c_occ_slice = MatrixFull::new([nao, nocc], 0.0);
        for mu in 0..nao {
            for col in 0..nocc {
                c_occ_slice[[mu, col]] = c_mo[[mu, start_mo + col]];
            }
        }

        let mut h1_mo = MatrixFull::new([nmo, nocc], 0.0);
        _dgemm_full(&tmp, 'N', &c_occ_slice, 'N', &mut h1_mo, 1.0, 0.0);

        // Flatten to column-major vec
        let mut flat = vec![0.0; nmo * nocc];
        for col in 0..nocc {
            for row in 0..nmo {
                flat[row + col * nmo] = h1_mo[[row, col]];
            }
        }
        results.push(flat);
    }

    results
}

/// Build overlap derivative integrals for atom `ia`.
///
/// Returns Vec of 3 [nao, nao] matrices in column-major flat format,
/// one per Cartesian direction (x, y, z).
pub fn build_s1ao_deriv(mol: &crate::Molecule, ia: usize) -> Vec<Vec<f64>> {
    let nao = mol.num_basis;
    let nao2 = nao * nao;
    let aoslices = mol.aoslice_by_atom();
    let p0 = aoslices[ia][2] as usize;
    let p1 = aoslices[ia][3] as usize;

    let cint = mol.initialize_cint(false);

    // int1e_ipovlp: overlap first derivative, shape [3, nao, nao]
    let (ipovlp, _): (Vec<f64>, Vec<usize>) =
        cint.integrate_row_major("int1e_ipovlp", "s1", None).into();

    // Build s1 matching PySCF's hess_elec:
    //   int1e_ipovlp returns ∂S/∂x summed over ALL atoms
    //   s1a[x,i,j] = -∂S_ij/∂x  (derivative wrt i's center)
    //   We must only take atom ia's rows, then add the transpose.
    let mut s1 = vec![vec![0.0; nao2]; 3];
    for x in 0..3 {
        // Rows: derivative on atom ia's AOs: s1ao[:,p0:p1] += s1a[:,p0:p1]
        for i in p0..p1 { for j in 0..nao {
            s1[x][i * nao + j] = -ipovlp[x * nao2 + i * nao + j];
        }}
        // Columns: transpose: s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        for i in 0..nao { for j in p0..p1 {
            s1[x][i * nao + j] += -ipovlp[x * nao2 + j * nao + i];
        }}
    }
    s1
}

/// Transform s1ao (one atom, 3 [nao,nao] matrices) to MO (nmo, nocc).
pub fn transform_s1ao_ao2mo(
    solver: &CPHFSolverPySCF,
    s1ao: &[Vec<f64>],  // 3 vectors, each length nao*nao
) -> Vec<Vec<f64>> {
    let nao = solver.nao;
    let nmo = solver.nmo;
    let nocc = solver.nocc;
    let start_mo = solver.start_mo;
    let c_mo = &solver.c_mo;

    let mut results = Vec::with_capacity(3);

    for x in 0..3 {
        // Build [nao, nao] matrix from flat column-major
        let mut s1_block = MatrixFull::new([nao, nao], 0.0);
        for i in 0..nao { for j in 0..nao {
            s1_block[[i, j]] = s1ao[x][i * nao + j];
        }}

        // tmp = C^T @ s1_block
        let mut tmp = MatrixFull::new([nao, nmo], 0.0);
        _dgemm_full(c_mo, 'T', &s1_block, 'N', &mut tmp, 1.0, 0.0);

        // c_occ_slice
        let mut c_occ_slice = MatrixFull::new([nao, nocc], 0.0);
        for mu in 0..nao { for col in 0..nocc {
            c_occ_slice[[mu, col]] = c_mo[[mu, start_mo + col]];
        }}

        let mut s1_mo = MatrixFull::new([nmo, nocc], 0.0);
        _dgemm_full(&tmp, 'N', &c_occ_slice, 'N', &mut s1_mo, 1.0, 0.0);

        let mut flat = vec![0.0; nmo * nocc];
        for col in 0..nocc { for row in 0..nmo {
            flat[row + col * nmo] = s1_mo[[row, col]];
        }}
        results.push(flat);
    }
    results
}
