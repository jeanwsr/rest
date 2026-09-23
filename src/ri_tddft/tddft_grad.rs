//! Analytic nuclear gradients of TDDFT/TDA excitation energies.
//!
//! Faithful port of PySCF `pyscf/grad/tdrks.py` (`_contract_xc_kernel`,
//! `grad_elec`) and `pyscf/grad/tdrhf.py` (`grad_elec`, CPHF Z-vector,
//! Pulay term) to REST's RI representation.
//!
//! The excited-state gradient is assembled as
//! `de_TDDFT = de_GS + response`, where `de_GS` is REST's validated
//! ground-state RKS gradient and `response = assemble(x,y) - assemble(0,0)`
//! is the amplitude-dependent part.
//!
//! PySCF calls two different `get_jk`:
//!   * `mf.get_jk(mol, dm, hermi=0)` returns the ordinary 0th-order J/K AO
//!     matrices (used for `veff0doo`/`veff0mop`/`veff0mom`);
//!   * `td_grad.get_jk(mol, dm)` returns derivative tables
//!     `vj[t,mu,nu] = -sum_{ls} d/dR_mu (mu nu|ls) D_ls` and
//!     `vk[t,mu,nu] = -sum_{ls} d/dR_mu (mu l|nu s) D_ls` (used for `veff1`).
//! Both are reproduced here in the RI approximation; the auxiliary-basis and
//! metric responses are added per bilinear form from the full potential
//! derivative (`RawRiTensors::d_i_from_blocks` + `d_j_atom`).

use crate::dft::num_int::{eval_ao_batch, eval_rho5_batch};
use crate::dft::libxc_itrf::eval_xc_eff;
use crate::dft::response::{response_ao_core, FxcHessianCache, VindWorkspace};
use crate::dft::xc_deriv::XCType;
use crate::grad::rhf::generator_deriv_hcore;
use crate::ri_cphf::CPHFSolverPySCF;
use crate::ri_gw::gw_grad::{build_raw_ri_tensors, AtomDerivBlocks, RawRiTensors};
use crate::ri_tddft::utils::tddft_occupation_parameters;
use crate::scf_io::SCF;
use libxc::prelude::*;
use rest_libcint::prelude::CInt;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dinverse};
use rest_tensors::matrix::matrixfullslice::MatrixFullSlice;
use rest_tensors::{MatrixFull, RIFull};
use rstsr::prelude::*;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;

/// Dense AO matrix, flat column-major `[nao, nao]`.
#[derive(Clone)]
struct AOMat {
    nao: usize,
    data: Vec<f64>,
}

impl AOMat {
    fn zeros(nao: usize) -> Self {
        AOMat { nao, data: vec![0.0; nao * nao] }
    }
    fn from_matrixfull(m: &MatrixFull<f64>) -> Self {
        AOMat { nao: m.size[0], data: m.data.clone() }
    }
    fn to_matrixfull(&self) -> MatrixFull<f64> {
        MatrixFull::from_vec([self.nao, self.nao], self.data.clone()).unwrap()
    }
    #[inline]
    fn get(&self, mu: usize, nu: usize) -> f64 {
        self.data[mu + nu * self.nao]
    }
    #[inline]
    fn set(&mut self, mu: usize, nu: usize, v: f64) {
        self.data[mu + nu * self.nao] = v;
    }
    fn add(&mut self, o: &AOMat) {
        for (a, b) in self.data.iter_mut().zip(o.data.iter()) {
            *a += b;
        }
    }
}

/// One rank factor pair `D += l r^T` (`l`,`r` are `[nao, nfac]` col-major).
#[derive(Clone)]
struct RankFactor {
    l: Vec<f64>,
    r: Vec<f64>,
    nfac: usize,
}

/// An AO density with its explicit rank-factorisation `D = sum l r^T` (used by
/// the exchange contractions, which need the factor structure).
struct Density {
    mat: AOMat,
    factors: Vec<RankFactor>,
}

impl Density {
    fn from_mat(mat: AOMat) -> Self {
        let nao = mat.nao;
        Density {
            mat,
            factors: vec![RankFactor { l: vec![0.0; nao], r: vec![0.0; nao], nfac: 0 }],
        }
    }
}

/// Pre-evaluated AO batches for the whole grid, one entry per block.
///
/// `contract_xc_kernel` is called twice per gradient (the second time for
/// `fxcz1`, which needs the Z-vector), and both passes walk the same grid with
/// the same `ao_deriv`.  `eval_ao_batch` is ~30% of the response cost, so the
/// second pass is pure duplicated work.  Whether it is worth caching depends
/// entirely on the memory budget, so `build_ao_cache` only materialises this
/// when the AO tensor fits (see `REST_TDDFT_GRAD_AOCACHE_MB`).
struct AoBlocks {
    blocks: Vec<(usize, usize, RIFull<f64>)>,
}

impl AoBlocks {
    fn get(&self, g0: usize, g1: usize) -> Option<&RIFull<f64>> {
        self.blocks
            .iter()
            .find(|(a, b, _)| *a == g0 && *b == g1)
            .map(|(_, _, ao)| ao)
    }
}

/// The TDDFT analytic-gradient engine for one excited state.
pub struct TddftGradEngine<'a> {
    pub scf: &'a SCF,
    pub state: usize,
    pub singlet: bool,
    pub tda: bool,

    nao: usize,
    nmo: usize,
    nocc: usize,
    nvir: usize,
    natm: usize,
    start_mo: usize,
    lumo: usize,

    x: Vec<f64>,
    y: Vec<f64>,

    raw: RawRiTensors,
    /// Inverse of the RI metric `(P|Q)`, precomputed once.  Solving the metric
    /// with a fresh dense factorisation on every call is O(naux^3) and would
    /// dominate the whole response (there are `nfac * nao` solves per factor
    /// set in `k_potential`/`vk_bra_atom`).
    jinv: MatrixFull<f64>,
    /// full-range HF-exchange coefficient (hyb for ordinary hybrids)
    hyb: f64,
    /// short-range HF-exchange correction (RSH only, 0 otherwise)
    hyb_sr: f64,
    fxc_cache: FxcHessianCache,
    /// libcint handle for the grid AO tables.  `eval_ao_batch` (the legacy
    /// serial spherical transform) is 5-6x slower than libcint on the same
    /// grid, and the XC passes walk the grid twice, so the AO evaluation is
    /// the second largest item of the response after the grid contractions.
    cint: CInt,
    /// `REST_TDDFT_GRAD_AO_LEGACY=1` restores the pre-refactor `eval_ao_batch`
    /// path (regression cross-check only).
    ao_legacy: bool,
}

impl<'a> TddftGradEngine<'a> {
    pub fn new(
        scf: &'a SCF,
        state: usize,
        singlet: bool,
        tda: bool,
        x: Vec<f64>,
        y: Vec<f64>,
    ) -> Self {
        let (start_mo, _num_state, nocc, nvir, _homo, lumo) = tddft_occupation_parameters(scf);
        let nao = scf.mol.num_basis;
        let nmo = scf.eigenvalues[0].len();
        let natm = scf.mol.geom.nfree;
        let raw = build_raw_ri_tensors(scf);
        let is_hf = scf.mol.xc_data.dfa_compnt_scf.is_empty();
        let hyb = if is_hf { 1.0 } else { scf.mol.xc_data.dfa_hybrid_scf };
        let hyb_sr = match crate::ri_tddft::utils::rsh_exchange_coeffs(scf) {
            Some((_omega, _c_full, c_sr)) => {
                panic!(
                    "TDDFT gradient: range-separated hybrids are not yet supported \
                     (c_sr={:.6}); use an ordinary hybrid or pure functional",
                    c_sr
                );
            }
            None => 0.0,
        };
        let jinv = {
            let j2 = MatrixFull::from_vec([raw.naux, raw.naux], raw.j2.clone()).unwrap();
            _dinverse(&j2).expect("TDDFT grad: singular RI metric")
        };
        let fxc_cache = crate::dft::response::prepare_fxc_hessian_cache(scf);
        let cint = crate::ri_jk::util::get_cint_mol(&scf.mol);
        let ao_legacy = std::env::var("REST_TDDFT_GRAD_AO_LEGACY").is_ok();
        TddftGradEngine {
            scf, state, singlet, tda,
            nao, nmo, nocc, nvir, natm, start_mo, lumo,
            x, y, raw, jinv, hyb, hyb_sr, fxc_cache, cint, ao_legacy,
        }
    }

    // ---- grid AO ----------------------------------------------------------

    /// Grid AO values + derivatives, evaluated with libcint.
    ///
    /// Layout is identical to the legacy [`eval_ao_batch`] (`[nao, ng, ncomp]`
    /// column-major), so it is a drop-in replacement; only the last bits differ
    /// (verified max |Δ| = 1.4e-14 against the legacy path on the naphthalene
    /// grid).  `REST_TDDFT_GRAD_AO_LEGACY=1` restores the legacy evaluator.
    #[inline]
    fn ao_batch(&self, coords: &[[f64; 3]], ao_deriv: usize, ng: usize) -> RIFull<f64> {
        if self.ao_legacy {
            eval_ao_batch(&self.scf.mol, coords, ao_deriv, ng)
        } else {
            crate::dft::num_int::eval_ao_batch_libcint(&self.cint, self.nao, coords, ao_deriv, ng)
        }
    }

    // ---- MO coefficients --------------------------------------------------

    fn mo_block(&self, col0: usize, n: usize) -> Vec<f64> {
        let eig = &self.scf.eigenvectors[0];
        let nao = self.nao;
        let mut m = vec![0.0; nao * n];
        for j in 0..n {
            for i in 0..nao {
                m[i + j * nao] = eig[[i, col0 + j]];
            }
        }
        m
    }

    fn c_occ(&self) -> Vec<f64> {
        self.mo_block(self.start_mo, self.nocc)
    }
    fn c_vir(&self) -> Vec<f64> {
        self.mo_block(self.lumo, self.nvir)
    }

    // ---- RI helpers -------------------------------------------------------

    fn solve_metric(&self, rhs: &[f64]) -> Vec<f64> {
        // `jinv` is precomputed in `new`; `out[p] = sum_q jinv[p,q] rhs[q]`.
        let naux = self.raw.naux;
        let mut out = vec![0.0; naux];
        for (q, &r) in rhs.iter().enumerate().take(naux) {
            if r == 0.0 {
                continue;
            }
            let base = q * naux;
            for p in 0..naux {
                out[p] += self.jinv.data[base + p] * r;
            }
        }
        out
    }

    /// `G = jinv @ Y` for `Y` of shape `[naux, ncol]` (column-major), done as a
    /// single BLAS-3 GEMM instead of `ncol` separate O(naux^2) matvecs.
    ///
    /// The RHS is borrowed (no copy); the previous version materialised a
    /// throwaway `MatrixFull` on every call.
    fn solve_metric_batch(&self, y: &[f64], ncol: usize) -> Vec<f64> {
        let naux = self.raw.naux;
        let size = [naux, ncol];
        let ind = [1usize, naux];
        let rhs = MatrixFullSlice { size: &size, indicing: &ind, data: y };
        let mut gm = MatrixFull::new([naux, ncol], 0.0);
        _dgemm_full(&self.jinv, 'N', &rhs, 'N', &mut gm, 1.0, 0.0);
        gm.data
    }

    /// `rho[P] = sum_{mu,nu} i3[mu,nu,P] D[mu,nu]` then `c = j2^{-1} rho`.
    fn coulomb_coeff(&self, d: &AOMat) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.raw.naux;
        let i3 = &self.raw.i3;
        let mut rho = vec![0.0; naux];
        for p in 0..naux {
            let mut s = 0.0;
            let base = p * nao * nao;
            for nu in 0..nao {
                for mu in 0..nao {
                    s += i3[base + mu + nu * nao] * d.get(mu, nu);
                }
            }
            rho[p] = s;
        }
        self.solve_metric(&rho)
    }

    /// 0th-order Coulomb potential `J[D][mu,nu] = sum_P i3[mu,nu,P] c_D[P]`.
    fn j_potential(&self, d: &AOMat) -> AOMat {
        let nao = self.nao;
        let naux = self.raw.naux;
        let c = self.coulomb_coeff(d);
        let mut out = AOMat::zeros(nao);
        for p in 0..naux {
            let base = p * nao * nao;
            for k in 0..nao * nao {
                out.data[k] += self.raw.i3[base + k] * c[p];
            }
        }
        out
    }

    /// `Y[mu,P,i] = sum_lambda i3[mu,lambda,P] l[lambda,i]` for one coefficient
    /// block, returned as `[naux, nao, nfac]` col-major (element `(p,mu,i)` at
    /// `p + mu*naux + i*nao*naux`).
    ///
    /// Each auxiliary index owns a contiguous `[nao,nao]` plane of `i3`, so the
    /// previous implementation's `raw.i3.clone()` (a full `naux*nao^2` copy per
    /// call) and the per-plane `MatrixFull` rebuild were pure overhead: the
    /// plane is copied straight into the BLAS operand.
    fn exchange_y(&self, l: &[f64], nfac: usize) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.raw.naux;
        let i3 = &self.raw.i3;
        let lm = MatrixFull::from_vec([nao, nfac], l.to_vec()).unwrap();
        let mut yt = vec![0.0; naux * nao * nfac];
        let mut ip = MatrixFull::new([nao, nao], 0.0);
        let mut blk = MatrixFull::new([nao, nfac], 0.0);
        for p in 0..naux {
            let base = p * nao * nao;
            ip.data.copy_from_slice(&i3[base..base + nao * nao]);
            _dgemm_full(&ip, 'N', &lm, 'N', &mut blk, 1.0, 0.0);
            for i in 0..nfac {
                let src = &blk.data[i * nao..(i + 1) * nao];
                let base = p + i * nao * naux;
                for a in 0..nao {
                    yt[base + a * naux] = src[a];
                }
            }
        }
        yt
    }

    /// 0th-order exchange potential for `D = sum_factors l r^T`.
    ///
    /// With `yl`, `yr` in the `[naux,nao,nfac]` layout and `G = jinv @ yr`,
    /// the contraction is `K[mu,nu] += sum_i (yl_i G_i^T)[mu,nu]`, i.e. one
    /// BLAS-3 GEMM per factor instead of a four-deep scalar loop.  `yl` and `g`
    /// are re-packed into `[nao, naux*nfac]` (cheap: `nao*naux*nfac` moves vs
    /// `2*nao^2*naux*nfac` FLOPs).
    fn k_potential(&self, factors: &[RankFactor]) -> AOMat {
        let nao = self.nao;
        let naux = self.raw.naux;
        let mut out = MatrixFull::new([nao, nao], 0.0);
        for f in factors {
            if f.nfac == 0 {
                continue;
            }
            let yl = self.exchange_y(&f.l, f.nfac);
            let yr = self.exchange_y(&f.r, f.nfac);
            let g = self.solve_metric_batch(&yr, nao * f.nfac);
            let cols = naux * f.nfac;
            let mut a = vec![0.0; nao * cols];
            let mut b = vec![0.0; nao * cols];
            for i in 0..f.nfac {
                for mu in 0..nao {
                    let src = mu * naux + i * nao * naux;
                    let dst = mu + i * naux * nao;
                    for p in 0..naux {
                        a[dst + p * nao] = yl[src + p];
                        b[dst + p * nao] = g[src + p];
                    }
                }
            }
            let am = MatrixFull::from_vec([nao, cols], a).unwrap();
            let bm = MatrixFull::from_vec([nao, cols], b).unwrap();
            _dgemm_full(&am, 'N', &bm, 'T', &mut out, 1.0, 1.0);
        }
        AOMat::from_matrixfull(&out)
    }

    /// PySCF `vj` table for one atom/component: `-sum d/dR_mu (mu nu|ls) D_ls`,
    /// restricted to the first AO index `mu` of the atom.
    fn vj_bra_atom(&self, blocks: &AtomDerivBlocks, comp: usize, c: &[f64]) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.raw.naux;
        let nrow = blocks.nrow;
        let stride = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride..(comp + 1) * stride];
        let mut out = vec![0.0; nao * nao];
        for mu in blocks.row0..blocks.row0 + nrow {
            for nu in 0..nao {
                let mut s = 0.0;
                for p in 0..naux {
                    s += d1c[(mu - blocks.row0) + nu * nrow + p * nrow * nao] * c[p];
                }
                out[mu + nu * nao] = -s;
            }
        }
        out
    }

    /// Metric-solved exchange half-transform `G[:, nu, i] = jinv @ yr[:, nu, i]`
    /// for every factor of a rank-factor list.  This depends only on the
    /// generating density, **not** on the atom or the Cartesian component, so it
    /// is computed once per term instead of once per `(atom, component)`.
    fn k_g_list(&self, factors: &[RankFactor]) -> Vec<Vec<f64>> {
        factors
            .iter()
            .map(|f| {
                if f.nfac == 0 {
                    Vec::new()
                } else {
                    let yr = self.exchange_y(&f.r, f.nfac);
                    self.solve_metric_batch(&yr, self.nao * f.nfac)
                }
            })
            .collect()
    }

    /// PySCF `vk` table (first AO index only) for `D = sum l r^T`.
    ///
    /// `gs` holds the precomputed `k_g_list(factors)` entries (same order).
    fn vk_bra_atom(
        &self,
        blocks: &AtomDerivBlocks,
        comp: usize,
        factors: &[RankFactor],
        gs: &[Vec<f64>],
    ) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.raw.naux;
        let nrow = blocks.nrow;
        let stride = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride..(comp + 1) * stride];
        let mut out = vec![0.0; nao * nao];
        for (fi, f) in factors.iter().enumerate() {
            if f.nfac == 0 {
                continue;
            }
            let g = &gs[fi];
            // E[P,i] = sum_lambda d1c[(mu,lambda,P)] l[lambda,i]
            let lm = MatrixFull::from_vec([nao, f.nfac], f.l.clone()).unwrap();
            let mut d1mu = MatrixFull::new([nao, naux], 0.0);
            let mut e = MatrixFull::new([naux, f.nfac], 0.0);
            for mu in blocks.row0..blocks.row0 + nrow {
                for lam in 0..nao {
                    for p in 0..naux {
                        d1mu[[lam, p]] = d1c[(mu - blocks.row0) + lam * nrow + p * nrow * nao];
                    }
                }
                _dgemm_full(&d1mu, 'T', &lm, 'N', &mut e, 1.0, 0.0);
                let ed = &e.data;
                for nu in 0..nao {
                    let mut s = 0.0;
                    for i in 0..f.nfac {
                        let ebase = i * naux;
                        let gbase = (nu + i * nao) * naux;
                        for p in 0..naux {
                            s += ed[ebase + p] * g[gbase + p];
                        }
                    }
                    out[mu + nu * nao] -= s;
                }
            }
        }
        out
    }

    /// Full (AO+aux+metric) derivative of `J[D]` as `[nao,nao]`.
    fn vj_full_atom(&self, blocks: &AtomDerivBlocks, comp: usize, d: &AOMat) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.raw.naux;
        let di = self.raw.d_i_from_blocks(blocks, comp);
        let dj = self.raw.d_j_atom(blocks.atm, comp);
        let mut g = vec![0.0; naux];
        let mut dg = vec![0.0; naux];
        for p in 0..naux {
            let base = p * nao * nao;
            let mut s = 0.0;
            let mut ds = 0.0;
            for nu in 0..nao {
                for mu in 0..nao {
                    let k = base + mu + nu * nao;
                    s += self.raw.i3[k] * d.get(mu, nu);
                    ds += di[k] * d.get(mu, nu);
                }
            }
            g[p] = s;
            dg[p] = ds;
        }
        let c = self.solve_metric(&g);
        let djm = MatrixFull::from_vec([naux, naux], dj.clone()).unwrap();
        let cm = MatrixFull::from_vec([naux, 1], c.clone()).unwrap();
        let mut djc = MatrixFull::new([naux, 1], 0.0);
        _dgemm_full(&djm, 'N', &cm, 'N', &mut djc, 1.0, 0.0);
        let rhs: Vec<f64> = dg.iter().zip(djc.data.iter()).map(|(a, b)| a - b).collect();
        let dc = self.solve_metric(&rhs);
        let mut out = vec![0.0; nao * nao];
        for nu in 0..nao {
            for mu in 0..nao {
                let kk = mu + nu * nao;
                let mut s = 0.0;
                for p in 0..naux {
                    let k = kk + p * nao * nao;
                    s += di[k] * c[p] + self.raw.i3[k] * dc[p];
                }
                out[kk] = s;
            }
        }
        out
    }

    /// Full derivative of `K[D]`, `D = sum l r^T`, as `[nao,nao]`.
    fn vk_full_atom(&self, blocks: &AtomDerivBlocks, comp: usize, factors: &[RankFactor]) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.raw.naux;
        let di = self.raw.d_i_from_blocks(blocks, comp);
        let dj = self.raw.d_j_atom(blocks.atm, comp);
        let i3 = MatrixFull::from_vec([nao * nao, naux], self.raw.i3.clone()).unwrap();
        let div = MatrixFull::from_vec([nao * nao, naux], di.clone()).unwrap();
        let djm = MatrixFull::from_vec([naux, naux], dj.clone()).unwrap();
        let mut out = vec![0.0; nao * nao];
        for f in factors {
            if f.nfac == 0 {
                continue;
            }
            // yl/dyl: [P, nu, i]
            let build = |l: &[f64], src: &MatrixFull<f64>| -> Vec<f64> {
                let lm = MatrixFull::from_vec([nao, f.nfac], l.to_vec()).unwrap();
                let mut yt = vec![0.0; naux * nao * f.nfac];
                let mut ip = MatrixFull::new([nao, nao], 0.0);
                let mut blk = MatrixFull::new([nao, f.nfac], 0.0);
                for p in 0..naux {
                    for a in 0..nao {
                        for b in 0..nao {
                            ip[[a, b]] = src[[a + b * nao, p]];
                        }
                    }
                    _dgemm_full(&ip, 'N', &lm, 'N', &mut blk, 1.0, 0.0);
                    for i in 0..f.nfac {
                        for a in 0..nao {
                            yt[p + (a + i * nao) * naux] = blk[[a, i]];
                        }
                    }
                }
                yt
            };
            let yl = build(&f.l, &i3);
            let yr = build(&f.r, &i3);
            let dyl = build(&f.l, &div);
            let dyr = build(&f.r, &div);
            // G = jinv yr^T (here [P, nu, i])
            let mut g = yr.clone();
            for i in 0..f.nfac {
                for nu in 0..nao {
                    let rhs: Vec<f64> = (0..naux).map(|q| yr[q + (nu + i * nao) * naux]).collect();
                    let sol = self.solve_metric(&rhs);
                    for p in 0..naux {
                        g[p + (nu + i * nao) * naux] = sol[p];
                    }
                }
            }
            // G2 = jinv dJ G
            let gm = MatrixFull::from_vec([naux, nao * f.nfac], g.clone()).unwrap();
            let mut djg = MatrixFull::new([naux, nao * f.nfac], 0.0);
            _dgemm_full(&djm, 'N', &gm, 'N', &mut djg, 1.0, 0.0);
            let mut g2 = djg.data;
            for col in 0..nao * f.nfac {
                let rhs = &g2[col * naux..(col + 1) * naux];
                let sol = self.solve_metric(rhs);
                g2[col * naux..(col + 1) * naux].copy_from_slice(&sol);
            }
            for i in 0..f.nfac {
                for mu in 0..nao {
                    for nu in 0..nao {
                        let mut s = 0.0;
                        for p in 0..naux {
                            let kl = p + (mu + i * nao) * naux;
                            let kr = p + (nu + i * nao) * naux;
                            s += dyl[kl] * g[kr] + yl[kl] * dyr[kr] - yl[kl] * g2[kr];
                        }
                        out[mu + nu * nao] += s;
                    }
                }
            }
        }
        out
    }

    /// `(ao_deriv, ao_comp, blksize)` for the `contract_xc_kernel` grid loop.
    ///
    /// `ao_deriv = 2` (10 components) is required by the GGA helpers: the
    /// derivative tables of `gga_grad_sum` contract the second derivatives of
    /// the AO with the gradient of the weighted kernel, so the 6 components
    /// beyond the first derivative are genuinely needed (same as PySCF's
    /// `ao_deriv=2` GGA gradient path).
    ///
    /// The block size is derived from the memory that is actually **left** in
    /// the declared budget (`max_memory`), not from `max_memory` as if the
    /// process were empty.  The old rule
    /// (`max_memory / 8 / ((ao_comp+1) * nao)`) let a single grid block claim
    /// ~11/12 of a 2000 MB budget inside a process whose resident set already
    /// exceeded it, which is how the gradient acquired its 1.8 GB transient
    /// workspace.  See [`Self::xc_work_budget_bytes`].
    fn xc_grid_setup(&self, xc_type: XCType) -> (usize, usize, usize) {
        let nao = self.nao;
        let ngrids = self.scf.grids.as_ref().map(|g| g.weights.len()).unwrap_or(0);
        let ao_deriv = if xc_type == XCType::LDA { 1 } else { 2 };
        let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
        // Working set of one grid block: the AO tensor (twice — the evaluator
        // returns libcint's layout and the contraction kernels need the
        // transposed one, so both are alive during the conversion) plus the
        // `[nao, ng]`-sized helpers of the kernels (`eval_rho_response`'s A0
        // copy and D·A0 product, `xc_eval_mat`'s `aow_all`, `gga_grad_sum`'s
        // reusable `aow` scratch).
        let per_grid_bytes = 8.0 * nao as f64 * (2 * ao_comp + 2) as f64;
        let blksize = ((self.xc_work_budget_bytes() / per_grid_bytes) as usize)
            .min(ngrids)
            .max(4);
        (ao_deriv, ao_comp, blksize)
    }

    /// Bytes one pass of the XC grid loop may spend on its per-block workspace.
    ///
    /// `max_memory` is REST's declared process budget.  `REST_TDDFT_GRAD_XCBLK_MB`
    /// overrides the result; otherwise it is the head room left after the
    /// resident set, capped at 10% of `max_memory` so that a grid is always
    /// walked in several blocks (the block workspace then overlaps the resident
    /// data instead of adding to it) and floored so that a process which already
    /// exceeds its budget still runs.
    fn xc_work_budget_bytes(&self) -> f64 {
        const MIN_MB: f64 = 192.0;
        if let Ok(v) = std::env::var("REST_TDDFT_GRAD_XCBLK_MB") {
            if let Ok(mb) = v.parse::<f64>() {
                return mb.max(1.0) * 1.0e6;
            }
        }
        let max_memory = self.scf.mol.ctrl.max_memory.unwrap_or(2000.0);
        let used = crate::utilities::memory_batch::detect_used_memory_mb("proc");
        let avail = (max_memory - used).max(0.0);
        avail.min(0.1 * max_memory).max(MIN_MB) * 1.0e6
    }

    /// Pre-evaluate the AO for the whole grid **iff** it fits the cache budget.
    ///
    /// `contract_xc_kernel` runs twice per gradient with the same `ao_deriv`, so
    /// the second AO evaluation is duplicated work.  Caching removes that
    /// duplication but costs `8*nao*ngrids*ao_comp` bytes resident, which trades
    /// directly against the memory goal; the budget is therefore the same
    /// *available* head room the per-block workspace uses (override with
    /// `REST_TDDFT_GRAD_AOCACHE_MB`, disable with `REST_TDDFT_GRAD_NO_AOCACHE`).
    /// Since `eval_ao_batch_libcint` is 5-6x faster than the legacy evaluator,
    /// re-evaluating instead of caching is cheap, and large grids simply do not
    /// take the cache.
    fn build_ao_cache(&self, ao_deriv: usize, ao_comp: usize, blksize: usize) -> Option<AoBlocks> {
        if std::env::var("REST_TDDFT_GRAD_NO_AOCACHE").is_ok() {
            return None;
        }
        let grids = self.scf.grids.as_ref()?;
        let ngrids = grids.weights.len();
        let nao = self.nao;
        let bytes = (nao * ngrids * ao_comp * 8) as f64;
        // The cache trades memory for the second AO evaluation.  Since the
        // 2026 refactor `eval_ao_batch_libcint` makes that evaluation cheap
        // (~1.4 s for a 3156 MB deriv-2 grid tensor on naphthalene), so the
        // cache is only taken when it fits in the same *available* budget the
        // per-block workspace uses; large basis sets simply re-evaluate.
        // `REST_TDDFT_GRAD_AOCACHE_MB` overrides the budget.
        let budget = std::env::var("REST_TDDFT_GRAD_AOCACHE_MB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|mb| mb * 1_000_000.0)
            .unwrap_or_else(|| self.xc_work_budget_bytes());
        if bytes > budget {
            if std::env::var("REST_TDDFT_GRAD_TIME").is_ok() {
                eprintln!(
                    "[aocache] skipped: {:.0} MB > {:.0} MB budget (REST_TDDFT_GRAD_AOCACHE_MB to override)",
                    bytes / 1.0e6,
                    budget / 1.0e6
                );
            }
            return None;
        }
        let mut blocks = Vec::new();
        let mut g0 = 0usize;
        while g0 < ngrids {
            let g1 = (g0 + blksize).min(ngrids);
            let ao = self.ao_batch(&grids.coordinates[g0..g1], ao_deriv, g1 - g0);
            blocks.push((g0, g1, ao));
            g0 = g1;
        }
        Some(AoBlocks { blocks })
    }

    // ---- XC kernel contraction -------------------------------------------

    /// PySCF `_contract_xc_kernel`, returning `[4, nao, nao]` buffers
    /// `(f1vo, f1oo_k1ao, vxc1)`; the last two are `None` when not requested.
    ///
    /// `f1oo_k1ao` is the **combined** target `f1oo + 2*k1ao`.  PySCF keeps the
    /// two separately only because it assembles them separately; both of its
    /// uses are of the combination (`veff0doo` takes `f1oo + 2*k1ao`, and
    /// `veff1[1]` takes `2*(f1oo + fxcz1 + 2*k1ao)`).  Because the grid kernel
    /// is linear in the weighted kernel `wv`, folding `kxc` into the `fxc`
    /// contraction with weight 2 is exact and removes one full GGA grid kernel
    /// per call (5 -> 4 overall).
    #[allow(clippy::too_many_arguments)]
    fn contract_xc_kernel(
        &self,
        dmvo: Option<&AOMat>,
        dmoo: Option<&AOMat>,
        with_vxc: bool,
        with_kxc: bool,
        singlet: bool,
        ao_blocks: Option<&AoBlocks>,
        xc_setup: (usize, usize, usize),
    ) -> (Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>) {
        let nao = self.nao;
        let mut f1vo = vec![0.0; 4 * nao * nao];
        // `f1oo + 2*k1ao`
        let mut f1oo = if dmoo.is_some() { Some(vec![0.0; 4 * nao * nao]) } else { None };
        let mut v1ao = if with_vxc { Some(vec![0.0; 4 * nao * nao]) } else { None };

        let xc_data = &self.scf.mol.xc_data;
        let xc_type = if xc_data.use_density_gradient() { XCType::GGA } else { XCType::LDA };
        // libxc functionals without a third derivative (e.g. PBE) contribute
        // no `kxc` term; PySCF treats it as zero.
        // libxc builds used by REST may be configured with `XC_DONT_COMPILE_KXC`;
        // in that case `kxc` cannot be requested analytically and the KXC
        // contraction is reconstructed by differencing `fxc` (see
        // `contract_kxc_fd`).  `deriv` stays 2 then, and `kxc_raw` is `None`.
        let has_kxc = with_kxc
            && xc_data.dfa_compnt_scf.iter().all(|&code| {
                LibXCFunctional::from_number(code as _, LibXCSpin::Unpolarized).has_kxc()
            });
        let deriv = if has_kxc { 3 } else { 2 };

        let dmvo_is_zero = dmvo.map_or(true, |d| d.data.iter().all(|&v| v == 0.0));
        let dmoo_is_zero = dmoo.map_or(true, |d| d.data.iter().all(|&v| v == 0.0));
        // With no response density every `fxc`/`kxc` contraction is identically
        // zero, so only `exc`/`vxc` (deriv = 1) are needed.  This halves the
        // tensor work of the zero-amplitude assembly's grid pass.
        let deriv = if dmvo_is_zero && dmoo_is_zero { 1 } else { deriv };

        let dmvo_sym = dmvo.map(|d| {
            let mut o = AOMat::zeros(nao);
            for mu in 0..nao {
                for nu in 0..nao {
                    o.set(mu, nu, 0.5 * (d.get(mu, nu) + d.get(nu, mu)));
                }
            }
            o
        });

        let grids = self.scf.grids.as_ref().expect("DFT grids required");
        let ngrids = grids.weights.len();
        // The blocking is decided once per `assemble` call and handed in, so
        // that both XC passes and the pass-2 AO cache use identical block
        // boundaries (the budget reads the current RSS, which grows as the
        // gradient allocates).
        let (ao_deriv, ao_comp, blksize) = xc_setup;

        let mo_coeffs = vec![self.scf.eigenvectors[0].clone()];
        let occ = vec![self.scf.occupation[0].clone()];
        let nvar = if xc_type == XCType::GGA { 4 } else { 1 };

        let kxc_fd_eps = std::env::var("REST_TDDFT_GRAD_KXCFD_EPS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0e-6);

        let verbose_time = std::env::var("REST_TDDFT_GRAD_TIME").is_ok();
        let skip_kxc = std::env::var("REST_TDDFT_GRAD_NO_KXC").is_ok();
        let mut t_ao = 0.0f64;
        let mut t_rho = 0.0f64;
        let mut t_xc = 0.0f64;
        let mut t_resp = 0.0f64;
        let mut t_mat = 0.0f64;
        let mut t_kfd = 0.0f64;
        let mut _m = std::time::Instant::now();
        let mut g0 = 0usize;
        while g0 < ngrids {
            let g1 = (g0 + blksize).min(ngrids);
            let ng = g1 - g0;
            let coords = &grids.coordinates[g0..g1];
            let weights = &grids.weights[g0..g1];
            // Reuse the pre-evaluated block when the budget allowed caching it.
            let local_ao;
            let ao = match ao_blocks.and_then(|b| b.get(g0, g1)) {
                Some(a) => a,
                None => {
                    local_ao = self.ao_batch(coords, ao_deriv, ng);
                    &local_ao
                }
            };
            if verbose_time { t_ao += _m.elapsed().as_secs_f64(); _m = std::time::Instant::now(); }

            // ground-state rho for the libxc kernel
            let rho_tensor = eval_rho5_batch(&ao, xc_type, &mo_coeffs, &occ, 1, ng);
            let rho = {
                let raw = rho_tensor.raw();
                let off = rho_tensor.offset();
                raw[off..off + ng * nvar].to_vec()
            };
            if verbose_time { t_rho += _m.elapsed().as_secs_f64(); _m = std::time::Instant::now(); }
            let xc_tensors =
                eval_xc_eff(&xc_data.dfa_compnt_scf, &xc_data.dfa_paramr_scf, xc_type, 0, &rho, ng, deriv);
            if verbose_time { t_xc += _m.elapsed().as_secs_f64(); _m = std::time::Instant::now(); }
            let vxc_raw = tensor_slice(xc_tensors[1].as_ref().expect("vxc"), ng * nvar);
            let fxc_raw = if dmvo_is_zero && dmoo_is_zero {
                Vec::new()
            } else {
                tensor_slice(xc_tensors[2].as_ref().expect("fxc"), ng * nvar * nvar)
            };
            let kxc_raw = xc_tensors[3]
                .as_ref()
                .map(|k| tensor_slice(k, ng * nvar * nvar * nvar));
            let ao_data = &ao.data;
            // View the AO buffer once per grid block; `xc_eval_mat` is called
            // up to four times per block on the same `ao`.  `asarray` on a
            // reference is zero-copy — the old `ao.data.to_vec()` duplicated
            // the whole `[nao,ng,ao_comp]` block and pushed the real peak above
            // the `max_memory`-derived budget.
            let device = DeviceBLAS::default();
            let ao_ref = if xc_type == XCType::GGA {
                Some(rt::asarray((ao_data, [nao, ng, ao_comp].f(), &device)))
            } else {
                None
            };

            // `rho1` is needed by both the FXC (`f1vo`) and KXC (`f1oo`) parts;
            // form it once per block and share.
            let rho1 = if !dmvo_is_zero {
                dmvo_sym.as_ref().map(|d| eval_rho_response(ao_data, nao, ng, ao_comp, xc_type, d, 2.0))
            } else {
                None
            };
            if let (Some(rho1), true) = (rho1.as_ref(), !dmvo_is_zero) {
                let wv = contract_fxc(rho1, &fxc_raw, weights, ng, nvar);
                xc_eval_mat(&mut f1vo, ao_data, nao, ng, ao_comp, xc_type, &wv, ao_ref.as_ref().map(|t| t.view()));
            }
            // Combined `f1oo + 2*k1ao` target.
            let mut wv_oo: Option<Vec<f64>> = None;
            if !dmoo_is_zero {
                if let Some(dmoo) = dmoo {
                    let rho2 = eval_rho_response(ao_data, nao, ng, ao_comp, xc_type, dmoo, 2.0);
                    // singlet: unpolarized fxc directly (PySCF `_contract_xc_kernel`)
                    wv_oo = Some(contract_fxc(&rho2, &fxc_raw, weights, ng, nvar));
                }
            }
            if with_kxc && !skip_kxc {
                if let Some(rho1) = rho1.as_ref() {
                    let _kfd = std::time::Instant::now();
                    let wv_k = match kxc_raw.as_ref() {
                        Some(kxc_raw) => contract_kxc(rho1, kxc_raw, weights, ng, nvar),
                        None => contract_kxc_fd(
                            &xc_data.dfa_compnt_scf,
                            &xc_data.dfa_paramr_scf,
                            xc_type,
                            &rho,
                            rho1,
                            weights,
                            ng,
                            nvar,
                            kxc_fd_eps,
                        ),
                    };
                    if verbose_time { t_kfd += _kfd.elapsed().as_secs_f64(); }
                    match wv_oo.as_mut() {
                        Some(w) => for (a, b) in w.iter_mut().zip(wv_k.iter()) { *a += 2.0 * b; },
                        None => wv_oo = Some(wv_k.iter().map(|v| 2.0 * v).collect()),
                    }
                }
            }
            if let (Some(f1oo), Some(w)) = (f1oo.as_mut(), wv_oo.as_ref()) {
                xc_eval_mat(f1oo, ao_data, nao, ng, ao_comp, xc_type, w, ao_ref.as_ref().map(|t| t.view()));
            }
            if verbose_time { t_resp += _m.elapsed().as_secs_f64(); _m = std::time::Instant::now(); }
            if let (Some(v1ao), true) = (v1ao.as_mut(), with_vxc) {
                let wv = scale_weighted(&vxc_raw, weights, ng, nvar);
                xc_eval_mat(v1ao, ao_data, nao, ng, ao_comp, xc_type, &wv, ao_ref.as_ref().map(|t| t.view()));
            }
            if verbose_time { t_mat += _m.elapsed().as_secs_f64(); _m = std::time::Instant::now(); }
            g0 = g1;
        }

        if verbose_time {
            eprintln!(
                "[xckernel] ao={:.3} rho={:.3} eval_xc_eff={:.3} response={:.3} kxc_fd={:.3} eval_mat={:.3}",
                t_ao, t_rho, t_xc, t_resp, t_kfd, t_mat
            );
        }
        for m in [Some(&mut f1vo), f1oo.as_mut(), v1ao.as_mut()]
            .into_iter()
            .flatten()
        {
            for t in 0..3 {
                for v in m[(t + 1) * nao * nao..(t + 2) * nao * nao].iter_mut() {
                    *v *= -1.0;
                }
            }
        }
        (f1vo, f1oo, v1ao)
    }

    // ---- vresp -----------------------------------------------------------

    /// `mf.gen_response(singlet=None, hermi=1)` applied to a symmetric AO density
    /// with known rank factors.
    ///
    /// Unlike the CP-HF matvec (which starts from MO amplitudes), this must not
    /// project the density back onto the VO block: REST's MOs are orthonormal
    /// in the *overlap* metric, not the AO metric, so such a projection would
    /// not be exact. The response is built directly from the AO density.
    fn vresp_factored(&self, dm: &AOMat, factors: &[RankFactor]) -> AOMat {
        let mut v = self.j_potential(dm);
        if self.hyb != 0.0 {
            let k = self.k_potential(factors);
            for (a, b) in v.data.iter_mut().zip(k.data.iter()) {
                *a -= 0.5 * self.hyb * b;
            }
        }
        let fxc = crate::dft::response::compute_fxc_response_ao_cached(
            &self.fxc_cache,
            &dm.to_matrixfull(),
        );
        for (a, b) in v.data.iter_mut().zip(fxc.data.iter()) {
            *a += b;
        }
        v
    }

    // ---- main assembly ---------------------------------------------------

    /// Amplitude-dependent part of the excited-state gradient:
    /// `de_TDDFT = de_GS + response_gradient()`.
    ///
    /// By default this forms the response **directly** in one pass (PySCF's own
    /// assembly), instead of running the zero-amplitude assembly and
    /// subtracting.  `REST_TDDFT_GRAD_LEGACY=1` restores the old
    /// `assemble(x,y) - assemble(0,0)` oracle for regression checks.
    pub fn response_gradient(&self) -> MatrixFull<f64> {
        if std::env::var("REST_TDDFT_GRAD_LEGACY").is_ok() {
            self.response_gradient_legacy()
        } else {
            self.assemble(&self.x, &self.y, false, true)
        }
    }

    /// Expose a single assembly pass (diagnostics).
    pub fn debug_assembly(&self, zero: bool, resp_only: bool) -> MatrixFull<f64> {
        self.assemble(&self.x, &self.y, zero, resp_only)
    }

    /// Legacy oracle `assemble(x,y) - assemble(0,0)` (two full passes).  Kept
    /// for regression; mathematically equal to [`Self::response_gradient`].
    pub fn response_gradient_legacy(&self) -> MatrixFull<f64> {
        let full = self.assemble(&self.x, &self.y, false, false);
        let zero = self.assemble(&self.x, &self.y, true, false);
        let mut out = MatrixFull::new([3, self.natm], 0.0);
        for k in 0..out.data.len() {
            out.data[k] = full.data[k] - zero.data[k];
        }
        out
    }

    /// Electronic ground-state gradient as reproduced by the PySCF assembly at
    /// zero amplitude (diagnostic; should equal REST's `calc_rks` minus nuclear).
    pub fn zero_amplitude_gradient(&self) -> MatrixFull<f64> {
        self.assemble(&self.x, &self.y, true, false)
    }

    /// PySCF `tdrks.grad_elec` assembly.
    ///
    /// * `zero`      – evaluate at zero amplitude (diagnostic / legacy oracle).
    /// * `resp_only` – return only `assemble(x,y) - assemble(0,0)`, formed
    ///   algebraically.  The assembly is linear/bilinear in the response
    ///   densities, so the difference can be read off term by term:
    ///
    ///   | full assembly target            | response-only target |
    ///   |---------------------------------|----------------------|
    ///   | `D1 = 2*oo0 + dmz1doo`          | `dmz1doo`            |
    ///   | `dm1` OO block `doo + 2*I`      | `doo`                |
    ///   | `veff1[0]` contribution         | (cancels: identical) |
    ///   | `aux` targets `[D1, oo0, ...]`  | `[dmz1doo, oo0, ...]`|
    ///
    ///   `veff1[0]` follows only from `oo0` (whose derivative table and the
    ///   ground-state `vxc1` are the same in both assemblies) so its
    ///   contribution cancels exactly; it is still needed as the matrix
    ///   multiplying the `dmz1doo` target.  All other targets are zero at
    ///   zero amplitude, so their full value *is* the difference.
    fn assemble(&self, x: &[f64], y: &[f64], zero: bool, resp_only: bool) -> MatrixFull<f64> {
        let timing = std::env::var("REST_TDDFT_GRAD_TIME").is_ok();
        let mem_trace = std::env::var("REST_TDDFT_GRAD_MEM").is_ok();
        let mut _t0 = std::time::Instant::now();
        macro_rules! tick {
            ($label:expr) => {
                if timing {
                    eprintln!("[gradtime] {:<22} {:7.3}s", $label, _t0.elapsed().as_secs_f64());
                    _t0 = std::time::Instant::now();
                }
            };
        }
        macro_rules! memmark {
            ($label:expr) => {
                if mem_trace {
                    eprintln!(
                        "[gradmem] {:<22} rss={:8.1} MB  peak={:8.1} MB",
                        $label,
                        crate::hessian::memory_monitor::current_rss_mb(),
                        process_peak_rss_mb()
                    );
                }
            };
        }
        let nao = self.nao;
        let nocc = self.nocc;
        let nvir = self.nvir;
        let nmo = self.nmo;
        let (x, y) = if zero { (vec![0.0; x.len()], vec![0.0; y.len()]) } else { (x.to_vec(), y.to_vec()) };

        let c_occ = self.c_occ();
        let c_vir = self.c_vir();
        let c_all = &self.scf.eigenvectors[0];

        // Both the 0th-order and the derivative-stage exchange work is skipped
        // unless there is a non-zero HF-exchange fraction.  For pure
        // functionals `kf == 0`, so the whole K cost was previously paid and
        // then multiplied by zero.
        let kf = self.hyb;
        let need_k = kf != 0.0;

        // ---- densities (PySCF grad_elec step 1) ----
        let mut xpy = vec![0.0; nvir * nocc];
        let mut xmy = vec![0.0; nvir * nocc];
        for a in 0..nvir {
            for i in 0..nocc {
                xpy[a + i * nvir] = x[i + a * nocc] + y[i + a * nocc];
                xmy[a + i * nvir] = x[i + a * nocc] - y[i + a * nocc];
            }
        }
        let mut dvv = vec![0.0; nvir * nvir];
        let mut doo = vec![0.0; nocc * nocc];
        for a in 0..nvir {
            for b in 0..nvir {
                let mut s = 0.0;
                for i in 0..nocc {
                    s += xpy[a + i * nvir] * xpy[b + i * nvir] + xmy[a + i * nvir] * xmy[b + i * nvir];
                }
                dvv[a + b * nvir] = s;
            }
        }
        for i in 0..nocc {
            for j in 0..nocc {
                let mut s = 0.0;
                for a in 0..nvir {
                    s += xpy[a + i * nvir] * xpy[a + j * nvir] + xmy[a + i * nvir] * xmy[a + j * nvir];
                }
                doo[i + j * nocc] = -s;
            }
        }
        let dmxpy = build_dm(&c_vir, &xpy, nvir, &c_occ, nocc, nao);
        let dmxmy = build_dm(&c_vir, &xmy, nvir, &c_occ, nocc, nao);
        let dmzoo = {
            let a = build_dm(&c_occ, &doo, nocc, &c_occ, nocc, nao);
            let b = build_dm(&c_vir, &dvv, nvir, &c_vir, nvir, nao);
            let mut o = a;
            o.add(&b);
            o
        };

        // Rank factors for K (skipped entirely for pure functionals).
        //
        // Batch B: a transposed density needs no re-factorisation — `D = l r^T`
        // transposes to `(r, l)` with the *same* column count.  Building the
        // transpose via `C_o (X^T) C_v^T` used to inflate the rank from `nocc`
        // to `nvir`, which is expensive when `nvir >> nocc`.
        //
        // The virtual-space block is additionally written as
        // `C_v dvv C_v^T = U_+ U_+^T + U_- U_-^T` with `U_pm = C_v (X +/- Y)`,
        // i.e. `2*nocc` columns instead of `nvir`.  For TDA (`Y == 0`) the two
        // terms coincide (`U_+ == U_-`) and collapse into one factor scaled by 2.
        let y_is_zero = y.iter().all(|&v| v == 0.0);
        // `REST_TDDFT_GRAD_FACTOR_LEGACY=1` restores the pre-Batch-B
        // representation (re-factorised transposes with `nvir` columns,
        // `C_v dvv C_v^T` with `nvir` columns) for A/B comparison and as a
        // regression cross-check; see `tests/test_tddft_grad.rs`.
        let factor_legacy = std::env::var("REST_TDDFT_GRAD_FACTOR_LEGACY").is_ok();
        let f_dmxpy = if need_k { Some(rank_factor_product(&c_vir, &xpy, nvir, &c_occ, nocc, nao)) } else { None };
        let f_dmxmy = if need_k { Some(rank_factor_product(&c_vir, &xmy, nvir, &c_occ, nocc, nao)) } else { None };
        let f_doo = if need_k { Some(rank_factor_product(&c_occ, &doo, nocc, &c_occ, nocc, nao)) } else { None };
        let (f_dmxpy_t, f_dmxmy_t, f_dvv): (Option<RankFactor>, Option<RankFactor>, Vec<RankFactor>) =
            if !need_k {
                (None, None, Vec::new())
            } else if factor_legacy {
                (
                    Some(rank_factor_product(&c_occ, &transpose_sq(&xpy, nvir, nocc), nocc, &c_vir, nvir, nao)),
                    Some(rank_factor_product(&c_occ, &transpose_sq(&xmy, nvir, nocc), nocc, &c_vir, nvir, nao)),
                    vec![rank_factor_product(&c_vir, &dvv, nvir, &c_vir, nvir, nao)],
                )
            } else {
                let up = left_factor(&c_vir, &xpy, nvir, nocc, nao);
                let vv = if y_is_zero {
                    vec![scale_rank(&RankFactor { l: up.clone(), r: up, nfac: nocc }, 2.0)]
                } else {
                    let um = left_factor(&c_vir, &xmy, nvir, nocc, nao);
                    vec![
                        RankFactor { l: up.clone(), r: up, nfac: nocc },
                        RankFactor { l: um.clone(), r: um, nfac: nocc },
                    ]
                };
                (f_dmxpy.as_ref().map(transpose_rank), f_dmxmy.as_ref().map(transpose_rank), vv)
            };

        let dmb = Density {
            mat: {
                let mut m = dmxpy.clone();
                m.add(&transpose(&dmxpy, nao));
                m
            },
            factors: if need_k {
                vec![f_dmxpy.clone().unwrap(), f_dmxpy_t.clone().unwrap()]
            } else {
                Vec::new()
            },
        };
        let dmc = Density {
            mat: {
                let mut m = dmxmy.clone();
                let t = transpose(&dmxmy, nao);
                for (a, b) in m.data.iter_mut().zip(t.data.iter()) {
                    *a -= b;
                }
                m
            },
            factors: if need_k {
                vec![f_dmxmy.clone().unwrap(), neg_rank(&f_dmxmy_t.clone().unwrap())]
            } else {
                Vec::new()
            },
        };
        let dmzoo_d = Density {
            mat: dmzoo.clone(),
            factors: if need_k {
                let mut v = vec![f_doo.clone().unwrap()];
                v.extend(f_dvv.iter().cloned());
                v
            } else {
                Vec::new()
            },
        };

        tick!("setup+densities");
        memmark!("after setup");
        // ---- XC kernel contraction ----
        // `f1oo` is the combined `f1oo + 2*k1ao` target (see
        // `contract_xc_kernel`); the `kxc` part is folded in there and can be
        // suppressed with `REST_TDDFT_GRAD_NO_KXC`.
        // Pre-evaluate the grid AO once so the second (`fxcz1`) pass can reuse
        // it, when the memory budget allows (see `build_ao_cache`).
        let xc_type = if self.scf.mol.xc_data.use_density_gradient() { XCType::GGA } else { XCType::LDA };
        let (ao_deriv_c, ao_comp_c, blksize_c) = self.xc_grid_setup(xc_type);
        let ao_cache = self.build_ao_cache(ao_deriv_c, ao_comp_c, blksize_c);
        let (mut f1vo, mut f1oo, mut vxc1) =
            self.contract_xc_kernel(
                Some(&dmxpy),
                Some(&dmzoo),
                true,
                true,
                self.singlet,
                ao_cache.as_ref(),
                (ao_deriv_c, ao_comp_c, blksize_c),
            );
        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            for (nm, v) in [
                ("f1vo", Some(&f1vo)),
                ("vxc1", vxc1.as_ref()),
                ("f1oo_plus_2k1ao", f1oo.as_ref()),
            ] {
                if let Some(v) = v {
                    let tag = if zero { "zero" } else { "full" };
                    let mut f = std::fs::File::create(format!("{}/xc_{}_{}.bin", dir, nm, tag)).unwrap();
                    for x in v.iter() { f.write_all(&x.to_le_bytes()).unwrap(); }
                }
            }
        }
        if std::env::var("REST_TDDFT_GRAD_NO_XC").is_ok() {
            f1vo.iter_mut().for_each(|v| *v = 0.0);
            if let Some(m) = f1oo.as_mut() { m.iter_mut().for_each(|v| *v = 0.0); }
            if let Some(m) = vxc1.as_mut() { m.iter_mut().for_each(|v| *v = 0.0); }
        }
        let f1oo = f1oo.unwrap();
        let vxc1 = vxc1.unwrap();

        tick!("contract_xc_kernel");
        memmark!("after xc kernel #1");
        // exchange coefficient: `veff0doo = 2J - hyb*K` (both spin cases);
        // `veff` for the 1st (dmxpy) block drops the J term for triplets.
        let jf_dmb = if self.singlet { 2.0 } else { 0.0 };

        // ---- first get_jk: 0th-order potentials ----
        let vj0 = self.j_potential(&dmzoo);
        let vk0 = if need_k { self.k_potential(&dmzoo_d.factors) } else { AOMat::zeros(nao) };
        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            let tag = if zero { "zero" } else { "full" };
            let mut f = std::fs::File::create(format!("{}/vj0_{}.bin", dir, tag)).unwrap();
            for x in vj0.data.iter() { f.write_all(&x.to_le_bytes()).unwrap(); }
        }
        let mut veff0doo = AOMat::zeros(nao);
        for k in 0..nao * nao {
            veff0doo.data[k] = 2.0 * vj0.data[k] - kf * vk0.data[k] + f1oo[k];
        }
        let mut wvo = project_vo(&veff0doo, &c_occ, &c_vir, nao, nocc, nvir);
        for v in wvo.iter_mut() {
            *v *= 2.0;
        }
        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            let tag = if zero { "zero" } else { "full" };
            let mut f = std::fs::File::create(format!("{}/wvo0_{}.bin", dir, tag)).unwrap();
            for x in wvo.iter() {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }

        let vj1 = self.j_potential(&dmb.mat);
        let vk1 = if need_k { self.k_potential(&dmb.factors) } else { AOMat::zeros(nao) };
        let mut veff = AOMat::zeros(nao);
        for k in 0..nao * nao {
            veff.data[k] = jf_dmb * vj1.data[k] - kf * vk1.data[k]
                + f1vo[k] * if self.singlet { 2.0 } else { 1.0 };
        }
        let veff0mop = project_full(&veff, c_all, nao, nmo);
        apply_vo_coupling(&mut wvo, &veff0mop, &xpy, nocc, nvir, nmo, self.start_mo, self.lumo, -2.0, 2.0);

        // PySCF's pure-functional branch never forms `J[dmc]` at all (and its
        // `veffm` is `-K[dmc]`); the old `vj2` was dead code.
        let veffm = if need_k {
            let vk2 = self.k_potential(&dmc.factors);
            let mut m = AOMat::zeros(nao);
            for k in 0..nao * nao {
                m.data[k] = -kf * vk2.data[k];
            }
            m
        } else {
            AOMat::zeros(nao)
        };
        let veff0mom = project_full(&veffm, c_all, nao, nmo);
        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            let tag = if zero { "zero" } else { "full" };
            for (nm, v) in [("veff0mop", &veff0mop), ("veff0mom", &veff0mom)] {
                let mut f =
                    std::fs::File::create(format!("{}/{}_{}.bin", dir, nm, tag)).unwrap();
                for x in v.iter() {
                    f.write_all(&x.to_le_bytes()).unwrap();
                }
            }
        }
        apply_vo_coupling(&mut wvo, &veff0mom, &xmy, nocc, nvir, nmo, self.start_mo, self.lumo, -2.0, 2.0);

        tick!("first JK potentials");
        // ---- CPHF Z-vector ----
        let z1 = if zero || std::env::var("REST_TDDFT_GRAD_NO_CPHF").is_ok() {
            vec![0.0; nocc * nvir]
        } else {
            self.solve_zvector(&wvo)
        };
        // `z1` is in REST's VO layout `z1[i + a*nocc]`; `build_dm`/rank factors
        // need the `[nvir,nocc]` column-major matrix `z1[a + i*nvir]`.
        let z1_t = transpose_sq(&z1, nocc, nvir);
        let z1ao = build_dm(&c_vir, &z1_t, nvir, &c_occ, nocc, nao);
        let f_z1ao = if need_k { Some(rank_factor_product(&c_vir, &z1_t, nvir, &c_occ, nocc, nao)) } else { None };
        let f_z1ao_t = if factor_legacy {
            if need_k {
                Some(rank_factor_product(&c_occ, &transpose_sq(&z1_t, nvir, nocc), nocc, &c_vir, nvir, nao))
            } else {
                None
            }
        } else {
            f_z1ao.as_ref().map(transpose_rank)
        };
        let mut zsym = z1ao.clone();
        zsym.add(&transpose(&z1ao, nao));
        let veff_z = if need_k {
            self.vresp_factored(&zsym, &[f_z1ao.clone().unwrap(), f_z1ao_t.clone().unwrap()])
        } else {
            self.vresp_factored(&zsym, &[])
        };
        if std::env::var("REST_TDDFT_GRAD_NORMS").is_ok() {
            let n = |v: &[f64]| -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() };
            let fxc_resp = crate::dft::response::compute_fxc_response_ao_cached(
                &self.fxc_cache,
                &MatrixFull::from_vec([nao, nao], zsym.data.clone()).unwrap(),
            );
            let jnorm = n(&veff_z
                .data
                .iter()
                .zip(fxc_resp.data.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>());
            // cross-check the raw-RI J against REST's `compute_j_upper`
            let oo0_dbg = build_dm(&c_occ, &identity(nocc), nocc, &c_occ, nocc, nao);
            let j_raw = self.j_potential(&oo0_dbg);
            let j_up = crate::dft::response::compute_j_upper(
                self.scf,
                &vec![MatrixFull::from_vec([nao, nao], oo0_dbg.data.clone()).unwrap()],
            )
            .to_matrixfull()
            .unwrap();
            let jz_up = crate::dft::response::compute_j_upper(
                self.scf,
                &vec![MatrixFull::from_vec([nao, nao], zsym.data.clone()).unwrap()],
            )
            .to_matrixfull()
            .unwrap();
            println!(
                "[norms] zsym={:.6} jzsym={:.6} jnorm={:.6} fxcresp={:.6} veffz={:.6}",
                n(&zsym.data),
                n(&jz_up.data),
                jnorm,
                n(&fxc_resp.data),
                n(&veff_z.data),
            );
        }

        tick!("cphf solve");
        memmark!("after cphf");
        // ---- im0 ----
        let zeta = self.zeta_matrix();
        let mut im0_mo = vec![0.0; nmo * nmo];
        let mut tmp = veff0doo.clone();
        tmp.add(&veff_z);
        let oo = project_oo(&tmp, &c_occ, nao, nocc);
        for j in 0..nocc {
            for i in 0..nocc {
                im0_mo[(self.start_mo + i) + (self.start_mo + j) * nmo] = oo[i + j * nocc];
            }
        }
        for j in 0..nocc {
            for i in 0..nocc {
                let mut s = 0.0;
                for a in 0..nvir {
                    s += veff0mop[(self.lumo + a) + (self.start_mo + i) * nmo] * xpy[a + j * nvir]
                        + veff0mom[(self.lumo + a) + (self.start_mo + i) * nmo] * xmy[a + j * nvir];
                }
                im0_mo[(self.start_mo + i) + (self.start_mo + j) * nmo] += s;
            }
        }
        // vv block: im0[a,c] = sum_i veff0mop[c,i] xpy[a,i] (+ mom term)
        for c in 0..nvir {
            for a in 0..nvir {
                let mut s = 0.0;
                for i in 0..nocc {
                    s += veff0mop[(self.lumo + c) + (self.start_mo + i) * nmo] * xpy[a + i * nvir]
                        + veff0mom[(self.lumo + c) + (self.start_mo + i) * nmo] * xmy[a + i * nvir];
                }
                im0_mo[(self.lumo + a) + (self.lumo + c) * nmo] = s;
            }
        }
        // vo block: im0[a,k] = 2 sum_i veff0mop[k,i] xpy[a,i] (+ mom term)
        for k in 0..nocc {
            for a in 0..nvir {
                let mut s = 0.0;
                for i in 0..nocc {
                    s += veff0mop[(self.start_mo + k) + (self.start_mo + i) * nmo] * xpy[a + i * nvir]
                        + veff0mom[(self.start_mo + k) + (self.start_mo + i) * nmo] * xmy[a + i * nvir];
                }
                im0_mo[(self.lumo + a) + (self.start_mo + k) * nmo] = 2.0 * s;
            }
        }
        // In the response-only form the OO block of `dm1` must not carry the
        // ground-state `2*I`, which makes `im0` the difference `im0^full -
        // im0^zero` directly.
        let mut dm1 = vec![0.0; nmo * nmo];
        for j in 0..nocc {
            for i in 0..nocc {
                dm1[(self.start_mo + i) + (self.start_mo + j) * nmo] =
                    doo[i + j * nocc] + if i == j && !resp_only { 2.0 } else { 0.0 };
            }
        }
        for b in 0..nvir {
            for a in 0..nvir {
                dm1[(self.lumo + a) + (self.lumo + b) * nmo] = dvv[a + b * nvir];
            }
        }
        for i in 0..nocc {
            for a in 0..nvir {
                dm1[(self.lumo + a) + (self.start_mo + i) * nmo] = z1[i + a * nocc];
            }
        }
        let mut acc = vec![0.0; nmo * nmo];
        for k in 0..nmo * nmo {
            acc[k] = im0_mo[k] + zeta[k] * dm1[k];
        }
        let im0_ao = transform_mo_to_ao(&c_occ, &c_vir, acc.as_slice(), nmo, nocc, nvir, self.start_mo, self.lumo, nao);

        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            let tag = if zero { "zero" } else { "full" };
            let mut f = std::fs::File::create(format!("{}/im0ao_{}.bin", dir, tag)).unwrap();
            for v in &im0_ao.data { f.write_all(&v.to_le_bytes()).unwrap(); }
        }

        tick!("im0");
        // ---- second get_jk: derivative tables ----
        let oo0 = build_dm(&c_occ, &identity(nocc), nocc, &c_occ, nocc, nao);
        let f_oo0 = if need_k { Some(rank_factor_product(&c_occ, &identity(nocc), nocc, &c_occ, nocc, nao)) } else { None };
        let dmz1doo = {
            let mut m = z1ao.clone();
            m.add(&dmzoo);
            m
        };
        if std::env::var("REST_TDDFT_GRAD_NORMS").is_ok() {
            let n = |v: &[f64]| -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() };
            println!("[norms2] im0ao={:.6} oo0={:.6} dmz1doo={:.6}",
                n(&im0_ao.data), n(&oo0.data), n(&dmz1doo.data));
        }
        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            let tag = if zero { "zero" } else { "full" };
            for (nm, v) in [("z1", &z1), ("z1ao", &z1ao.data), ("dmzoo", &dmzoo.data), ("dmz1doo", &dmz1doo.data), ("dmxpy", &dmxpy.data), ("wvo", &wvo), ("veff0doo", &veff0doo.data), ("veff_z", &veff_z.data)] {
                let nm = format!("{}_{}", nm, tag);
                let mut f = std::fs::File::create(format!("{}/{}.bin", dir, nm)).unwrap();
                for x in v.iter() { f.write_all(&x.to_le_bytes()).unwrap(); }
            }
            let mut f = std::fs::File::create(format!("{}/meta_{}.txt", dir, tag)).unwrap();
            writeln!(f, "nao={} nocc={} nvir={} nmo={}", nao, nocc, nvir, nmo).unwrap();
        }
        // generating density `dmB = dmz1doo + dmz1doo^T`
        let dm1sym = {
            let mut m = dmz1doo.clone();
            let t = transpose(&dmz1doo, nao);
            m.add(&t);
            m
        };
        // `dm1sym = dmz1doo + dmz1doo^T = z1ao + z1ao^T + 2*dmzoo`, so the
        // `dmzoo` rank factors must be counted **twice** (the `oo0` term above
        // uses the same `f_doo`/`f_dvv` once).  Getting this wrong leaves the K
        // derivative table for `dm1sym` too small and is invisible for pure
        // functionals (`hyb == 0`).
        let fac_b = if need_k {
            if factor_legacy {
                vec![
                    f_z1ao.clone().unwrap(),
                    f_z1ao_t.clone().unwrap(),
                    scale_rank(&f_doo.clone().unwrap(), 2.0),
                    scale_rank(&f_dvv[0].clone(), 2.0),
                ]
            } else {
                let mut v = vec![
                    f_z1ao.clone().unwrap(),
                    f_z1ao_t.clone().unwrap(),
                    scale_rank(&f_doo.clone().unwrap(), 2.0),
                ];
                v.extend(f_dvv.iter().map(|f| scale_rank(f, 2.0)));
                v
            }
        } else {
            Vec::new()
        };

        // PySCF second `get_jk` densities: (oo0, dmz1doo+dmz1doo^T, dmxpy+dmxpy^T, dmxmy-dmxmy^T)
        let gen_mats: [&AOMat; 4] = [&oo0, &dm1sym, &dmb.mat, &dmc.mat];
        let gen_factors: [Vec<RankFactor>; 4] = if need_k {
            [vec![f_oo0.clone().unwrap()], fac_b, dmb.factors.clone(), dmc.factors.clone()]
        } else {
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()]
        };
        // veff1 coefficients per density index (PySCF):
        //   0: -vk+2vj  1: -vk+2vj  2: singlet -vk+2vj / triplet -vk  3: -vk
        let cj = [2.0, 2.0, if self.singlet { 2.0 } else { 0.0 }, 0.0];
        let ck = [-kf, -kf, -kf, -kf];
        let need_j = [cj[0] != 0.0, cj[1] != 0.0, cj[2] != 0.0, cj[3] != 0.0];
        let need_k_i = [need_k, need_k, need_k, need_k];

        // Targets and outer coefficients of the four bilinear forms.  In the
        // response-only form the first target is the response density directly.
        let d1_full_mat = if resp_only {
            dmz1doo.clone()
        } else {
            let mut m = oo0.clone();
            for v in m.data.iter_mut() {
                *v *= 2.0;
            }
            m.add(&dmz1doo);
            m
        };
        let aux_targets: [&AOMat; 4] = [&d1_full_mat, &oo0, &dmxpy, &dmxmy];
        // Per-term weights of the auxiliary-basis/metric response.  Calibrated
        // against PySCF's DF response of the same four bilinear forms
        // (`bench/tddft_grad_aux_target.py` + `bench/fit_aux_weights.py`): the
        // residual against the reference per-atom response is <= 4e-5 a.u. for
        // H2O/def2-svp/PBE/TDA.  Overridable for re-calibration.
        let aux_coeff = std::env::var("REST_TDDFT_GRAD_AUXCOEFF")
            .ok()
            .and_then(|v| {
                let p: Vec<f64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if p.len() == 4 { Some([p[0], p[1], p[2], p[3]]) } else { None }
            })
            .unwrap_or([0.0, 2.0, 2.0, 0.0]);

        // RI auxiliary-basis/metric response forces (per atom, per term),
        // accumulated in the same per-atom pass as the AO-bra tables so the
        // `int3c2e_ip1` and `int3c2e_ip2` blocks are generated once.
        let want_aux = std::env::var("REST_TDDFT_GRAD_NO_AUX").is_err();
        let mut aux = vec![[0.0f64; 3]; self.natm * 4];
        let c_gens: Vec<Vec<f64>> =
            (0..4).map(|i| if want_aux { self.coulomb_coeff(gen_mats[i]) } else { Vec::new() }).collect();
        let c_tars: Vec<Vec<f64>> =
            (0..4).map(|i| if want_aux { self.coulomb_coeff(aux_targets[i]) } else { Vec::new() }).collect();
        let gsl: Vec<Vec<Vec<f64>>> = if need_k {
            (0..4).map(|i| self.k_g_list(&gen_factors[i])).collect()
        } else {
            Vec::new()
        };
        let cs: Vec<Vec<f64>> =
            (0..4).map(|i| if need_j[i] { self.coulomb_coeff(gen_mats[i]) } else { Vec::new() }).collect();

        let mut vj_tab: Vec<Vec<f64>> = vec![vec![0.0; 3 * nao * nao]; 4];
        let mut vk_tab: Vec<Vec<f64>> = vec![vec![0.0; 3 * nao * nao]; 4];
        let mut t_blocks = 0.0f64;
        let mut t_vj = 0.0f64;
        let mut t_vk = 0.0f64;
        let mut t_aux = 0.0f64;
        for atm in 0..self.natm {
            let _tb = std::time::Instant::now();
            let blocks = self.raw.d_atom_blocks(atm);
            t_blocks += _tb.elapsed().as_secs_f64();
            for comp in 0..3 {
                for i in 0..4 {
                    if need_j[i] {
                        let _t = std::time::Instant::now();
                        let t = self.vj_bra_atom(&blocks, comp, &cs[i]);
                        for k in 0..nao * nao {
                            vj_tab[i][comp * nao * nao + k] += t[k];
                        }
                        t_vj += _t.elapsed().as_secs_f64();
                    }
                    if need_k_i[i] {
                        let _t = std::time::Instant::now();
                        let tk = self.vk_bra_atom(&blocks, comp, &gen_factors[i], &gsl[i]);
                        for k in 0..nao * nao {
                            vk_tab[i][comp * nao * nao + k] += tk[k];
                        }
                        t_vk += _t.elapsed().as_secs_f64();
                    }
                }
                if want_aux {
                    let _t = std::time::Instant::now();
                    let mut aux_atm = [[0.0f64; 3]; 4];
                    self.aux_forces_atom(
                        &blocks, comp, &c_gens, &c_tars, &gen_mats, &aux_targets, &aux_coeff, &cj, &ck,
                        &mut aux_atm,
                    );
                    for i in 0..4 {
                        aux[atm * 4 + i][comp] += aux_atm[i][comp];
                    }
                    t_aux += _t.elapsed().as_secs_f64();
                }
            }
        }
        if timing {
            eprintln!(
                "[jkbreak] blocks={:.3} vj={:.3} vk={:.3} aux={:.3}",
                t_blocks, t_vj, t_vk, t_aux
            );
        }
        if let Ok(dir) = std::env::var("REST_TDDFT_GRAD_DUMP") {
            use std::io::Write;
            let tag = if zero { "zero" } else { "full" };
            for i in 0..4 {
                for (nm, tab) in [("vjtab", &vj_tab[i]), ("vktab", &vk_tab[i])] {
                    let mut f =
                        std::fs::File::create(format!("{}/{}_{}_{}.bin", dir, nm, i, tag)).unwrap();
                    for x in tab.iter() {
                        f.write_all(&x.to_le_bytes()).unwrap();
                    }
                }
            }
            for (nm, v) in [("vk0", &vk0), ("vk1", &vk1)] {
                let mut f = std::fs::File::create(format!("{}/{}_{}.bin", dir, nm, tag)).unwrap();
                for x in v.data.iter() {
                    f.write_all(&x.to_le_bytes()).unwrap();
                }
            }
            // raw bilinear-form aux forces, per (term, atom, component): dump
            // for calibration of the per-term weights against PySCF.
            let mut line = String::new();
            for i in 0..4 {
                for atm in 0..self.natm {
                    for c in 0..3 {
                        line.push_str(&format!("{} {} {} {:.16e}\n", i, atm, c, aux[atm * 4 + i][c]));
                    }
                }
            }
            std::fs::write(format!("{}/auxforce_{}.txt", dir, tag), line).unwrap();
        }

        let mut veff1: Vec<Vec<f64>> = vec![vec![0.0; 3 * nao * nao]; 4];
        for i in 0..4 {
            for k in 0..3 * nao * nao {
                veff1[i][k] = cj[i] * vj_tab[i][k] + ck[i] * vk_tab[i][k];
            }
        }
        // vxc1 derivative -> veff1[0].  This term is identical in the full and
        // zero assemblies, so in the response-only form it cancels; it is still
        // added because `veff1[0]` multiplies the `dmz1doo` target below.
        for k in 0..3 * nao * nao {
            veff1[0][k] += vxc1[nao * nao + k];
        }
        // f1oo + fxcz1 + 2 k1ao -> veff1[1] (*2)
        // At zero amplitude `z1ao == 0`, so `fxcz1` is zero; skip the entire
        // (AO-batch dominated) grid pass in that assembly.
        let (fxcz1, _, _) = if zero || z1ao.data.iter().all(|&v| v == 0.0) {
            (vec![0.0; 4 * nao * nao], None, None)
        } else {
            self.contract_xc_kernel(
                Some(&z1ao),
                None,
                false,
                false,
                true,
                ao_cache.as_ref(),
                (ao_deriv_c, ao_comp_c, blksize_c),
            )
        };
        for k in 0..3 * nao * nao {
            veff1[1][k] += 2.0 * (f1oo[nao * nao + k] + fxcz1[nao * nao + k]);
        }
        // 2 f1vo -> veff1[2]
        if self.singlet {
            for k in 0..3 * nao * nao {
                veff1[2][k] += 2.0 * f1vo[nao * nao + k];
            }
        }

        tick!("second get_jk tables");
        memmark!("after xc kernel #2");
        // ---- final per-atom assembly ----
        let ao_slice = self.scf.mol.aoslice_by_atom();
        let mut hcore_gen = generator_deriv_hcore(self.scf);
        let s1 = self.overlap_deriv();
        let mut de = MatrixFull::new([3, self.natm], 0.0);

        // per-term diagnostics: accumulate the total over all atoms so the
        // translational sum rule can be checked term by term.
        let terms_dbg = std::env::var("REST_TDDFT_GRAD_TERMS").is_ok();
        let mut term_tot = [[0.0f64; 3]; 6]; // hcore, veff0sym, pulay, v1, v2, v3

        for atm in 0..self.natm {
            let [_, _, p0, p1] = ao_slice[atm];
            let h1 = hcore_gen(atm); // [nao,nao,3]

            // Target of the hcore/`veff1[0]` contraction: `D1` normally, the
            // response density `dmz1doo` in the response-only form.
            let d1 = &d1_full_mat;

            let mut e1 = [0.0f64; 3];
            // hcore_deriv . D1
            for t in 0..3 {
                let mut s = 0.0;
                for nu in 0..nao {
                    for mu in 0..nao {
                        s += h1[[mu, nu, t]] * d1.get(mu, nu);
                    }
                }
                e1[t] += s;
            }
            // sym(veff1[0]) . D1 : PySCF adds `veff1[0][:,blk]` to the rows and
            // `veff1[0][:,blk]^T` to the columns of h1ao, then contracts with D1.
            for t in 0..3 {
                let mut s = 0.0;
                for nu in 0..nao {
                    for mu in p0..p1 {
                        let v = veff1[0][t * nao * nao + mu + nu * nao];
                        s += v * d1.get(mu, nu);
                    }
                }
                for mu in 0..nao {
                    for nu in p0..p1 {
                        let vt = veff1[0][t * nao * nao + nu + mu * nao];
                        s += vt * d1.get(mu, nu);
                    }
                }
                e1[t] += s;
            }
            if terms_dbg {
                for t in 0..3 {
                    term_tot[0][t] += e1[t];
                }
            }
            // Pulay
            for t in 0..3 {
                let mut s = 0.0;
                for nu in 0..nao {
                    for mu in p0..p1 {
                        s -= s1[t * nao * nao + mu + nu * nao] * im0_ao.get(mu, nu);
                    }
                }
                for mu in 0..nao {
                    for nu in p0..p1 {
                        s -= s1[t * nao * nao + nu + mu * nao] * im0_ao.get(mu, nu);
                    }
                }
                e1[t] += s;
            }
            if terms_dbg {
                for t in 0..3 {
                    term_tot[1][t] += e1[t];
                }
            }
            // term1: rows(veff1[1]) . oo0
            for t in 0..3 {
                let mut s = 0.0;
                for nu in 0..nao {
                    for mu in p0..p1 {
                        s += veff1[1][t * nao * nao + mu + nu * nao] * oo0.get(mu, nu);
                    }
                }
                e1[t] += s;
            }
            if terms_dbg {
                for t in 0..3 {
                    term_tot[2][t] += e1[t];
                }
            }
            // term2: 2*(rows+cols)(veff1[2]) . dmxpy
            for t in 0..3 {
                let mut s = 0.0;
                for nu in 0..nao {
                    for mu in p0..p1 {
                        s += veff1[2][t * nao * nao + mu + nu * nao] * dmxpy.get(mu, nu);
                    }
                }
                for mu in 0..nao {
                    for nu in p0..p1 {
                        s += veff1[2][t * nao * nao + nu + mu * nao] * dmxpy.get(mu, nu);
                    }
                }
                e1[t] += 2.0 * s;
            }
            if terms_dbg {
                for t in 0..3 {
                    term_tot[3][t] += e1[t];
                }
            }
            // term3: 2*(rows-cols)(veff1[3]) . dmxmy
            for t in 0..3 {
                let mut s = 0.0;
                for nu in 0..nao {
                    for mu in p0..p1 {
                        s += veff1[3][t * nao * nao + mu + nu * nao] * dmxmy.get(mu, nu);
                    }
                }
                for mu in 0..nao {
                    for nu in p0..p1 {
                        s -= veff1[3][t * nao * nao + nu + mu * nao] * dmxmy.get(mu, nu);
                    }
                }
                e1[t] += 2.0 * s;
            }
            if terms_dbg {
                for t in 0..3 {
                    term_tot[4][t] += e1[t];
                }
            }
            // aux/metric response
            let mut aux_e1 = [0.0f64; 3];
            for l in 0..4 {
                for c in 0..3 {
                    let v = aux_coeff[l] * aux[atm * 4 + l][c];
                    e1[c] += v;
                    aux_e1[c] += v;
                }
            }
            if std::env::var("REST_TDDFT_GRAD_AUXDBG").is_ok() && atm < 3 {
                println!("[auxcontrib] atm{} {:?}", atm, aux_e1);
            }
            for t in 0..3 {
                de[[t, atm]] = e1[t];
            }
        }
        tick!("atom assembly+dump");
        memmark!("after assembly");
        if terms_dbg {
            let labels = ["h1", "pulay", "v1", "v2", "v3"];
            let mut tot = [0.0f64; 3];
            for (i, l) in labels.iter().enumerate() {
                let mut d = [0.0f64; 3];
                for t in 0..3 {
                    d[t] = if i == 0 { term_tot[0][t] } else { term_tot[i][t] - term_tot[i - 1][t] };
                    tot[t] += d[t];
                }
                println!("[termsum:{}] {:?}", l, d);
            }
            println!("[termsum:total] {:?}", tot);
        }
        de
    }

    fn solve_zvector(&self, wvo: &[f64]) -> Vec<f64> {
        use crate::ri_cphf::CPHFSolverPySCF;
        let solver = CPHFSolverPySCF::new_full(self.scf);
        // PySCF `cphf.solve` solves `(I + e_ai*fvind) z = -e_ai*wvo`, i.e. the
        // RHS must carry the orbital-energy denominators (`build_rhs_with_s1`).
        let rhs: Vec<f64> = wvo
            .iter()
            .zip(solver.e_ai.iter())
            .map(|(v, e)| -v * e)
            .collect();
        let out = solver.solve_krylov_batched(
            self.scf,
            Some(&self.fxc_cache),
            &[rhs],
            200,
            1.0e-10,
            1.0,
            1.0e-12,
        );
        out.into_iter().next().unwrap()
    }

    fn zeta_matrix(&self) -> Vec<f64> {
        let nmo = self.nmo;
        let e = &self.scf.eigenvalues[0];
        let mut z = vec![0.0; nmo * nmo];
        // PySCF: `zeta = direct_sum('i+j->ij', eps, eps)*.5` over the FULL MO
        // space; the ov/vo blocks are then overwritten.  The vv block (and any
        // frozen blocks) must keep the `0.5(eps_i+eps_j)` values.
        for i in 0..nmo {
            for j in 0..nmo {
                z[i + j * nmo] = 0.5 * (e[i] + e[j]);
            }
        }
        for a in 0..self.nvir {
            for i in 0..self.nocc {
                z[(self.lumo + a) + (self.start_mo + i) * nmo] = e[self.start_mo + i];
                z[(self.start_mo + i) + (self.lumo + a) * nmo] = e[self.lumo + a];
            }
        }
        z
    }

    fn overlap_deriv(&self) -> Vec<f64> {
        let mol_obj = &self.scf.mol;
        let mol = crate::ri_jk::util::get_cint_mol(mol_obj);
        let (out, _shape): (Vec<f64>, Vec<usize>) = mol.integrate("int1e_ipovlp", "s1", None).into();
        // PySCF `grad.rhf.get_ovlp` returns `-int1e_ipovlp`
        out.into_iter().map(|v| -v).collect()
    }

    /// RI auxiliary-basis/metric response forces for ONE atom and ONE
    /// Cartesian component, for the four bilinear forms (index `i`).
    ///
    /// The term coefficients `cj`/`ck` decide whether a form has an AO-space
    /// counterpart at all; `outer` carries the caller's per-term multiplicity
    /// (the calibrated `aux_coeff`) so forms whose outer weight is zero are
    /// skipped before any integral work.  This is a faithful extraction of the
    /// original `aux_forces` inner loop; see the calibration notes there.
    #[allow(clippy::too_many_arguments)]
    fn aux_forces_atom(
        &self,
        blocks: &AtomDerivBlocks,
        comp: usize,
        c_gens: &[Vec<f64>],
        c_tars: &[Vec<f64>],
        gen_mats: &[&AOMat; 4],
        targets: &[&AOMat; 4],
        outer: &[f64; 4],
        cj: &[f64; 4],
        ck: &[f64; 4],
        out: &mut [[f64; 3]; 4],
    ) {
        let nao = self.nao;
        let naux = self.raw.naux;
        let naux_a = blocks.naux_a;
        let stride2 = nao * nao * naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];
        for i in 0..4 {
            if cj[i] == 0.0 && ck[i] == 0.0 {
                continue;
            }
            if outer[i] == 0.0 {
                continue;
            }
            let gd = &gen_mats[i].data;
            let td = &targets[i].data;
            let c_gen = &c_gens[i];
            let c_tar = &c_tars[i];
            let mut acc = 0.0;
            for p_l in 0..naux_a {
                let p = blocks.aux0 + p_l;
                let base = p_l * nao * nao;
                let mut g_gen = 0.0;
                let mut g_tar = 0.0;
                for nu in 0..nao {
                    let col = nu * nao;
                    for mu in 0..nao {
                        let d = d2c[base + mu + col];
                        g_gen += gd[mu + col] * d;
                        g_tar += td[mu + col] * d;
                    }
                }
                let mut metric_tar = 0.0;
                let mut metric_gen = 0.0;
                for q in 0..naux {
                    let dj = self.raw.dj1[p + q * naux + comp * naux * naux];
                    metric_tar += dj * c_tar[q];
                    metric_gen += dj * c_gen[q];
                }
                // `int2c2e_ip1` differentiates only the first aux index; this
                // symmetric form is the calibration that reproduces REST's own
                // ground-state `de_jaux` exactly at zero amplitude
                // (see docs/TDDFT_解析梯度_实施方案.md).
                acc += -c_tar[p] * g_gen - c_gen[p] * g_tar
                    + c_gen[p] * metric_tar + c_tar[p] * metric_gen;
            }
            // match the AO contraction pattern of each term: term0 sym (full),
            // term1 rows-only (half), term2 rows+cols (full), term3 rows-cols
            // (half; zero for the symmetric J of an antisymmetric generator).
            let aux_scale = [1.0, 0.5, 1.0, 0.5];
            out[i][comp] += aux_scale[i] * acc;
        }
    }
}

// ---------------------------------------------------------------------------
// free helpers
// ---------------------------------------------------------------------------

fn tensor_slice(t: &Tsr<f64>, len: usize) -> Vec<f64> {
    let raw = t.raw();
    let off = t.offset();
    raw[off..off + len].to_vec()
}

/// Process-wide high-water RSS (`VmHWM`), in MiB; 0 when unavailable.
///
/// `VmHWM` is monotonic, so it only answers "did this stage push the peak
/// further up", never "what does the stage cost on its own"; the per-stage
/// `current_rss_mb()` marks printed next to it give the live footprint.
fn process_peak_rss_mb() -> f64 {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(kb) = rest.split_whitespace().next() {
                    if let Ok(v) = kb.parse::<f64>() {
                        return v / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

/// AO tensor element `(mu,g,c)` of an `eval_ao_batch` buffer `[nao,ng,ncomp]`
/// (column-major).
#[inline]
fn ao_val(ao: &[f64], nao: usize, ng: usize, mu: usize, g: usize, c: usize) -> f64 {
    ao[mu + g * nao + c * nao * ng]
}

/// `rho[g], rho_x[g], ...` of a **symmetric** AO density `dm` multiplied by
/// `scale` (mirrors `numint.eval_rho(..., hermi=1) * scale`).
///
/// PySCF's `_contract_xc_kernel` symmetrises `dmvo` first, and `dmoo` is
/// symmetric by construction, so the symmetric identity
/// `sum_mu A_d[mu] (D A_0)[mu] == sum_mu A_0[mu] (D A_d)[mu]` holds and the
/// spatial gradient needs only the single `D A_0` product
/// (`d_d rho = 2*s*sum_mu A_d[mu] (D A_0)[mu]`).  The previous version formed
/// four `D A_c` products per block.
fn eval_rho_response(
    ao: &[f64],
    nao: usize,
    ng: usize,
    _ao_comp: usize,
    xc_type: XCType,
    dm: &AOMat,
    scale: f64,
) -> Vec<f64> {
    let nvar = if xc_type == XCType::GGA { 4 } else { 1 };
    let dm_mat = MatrixFull::from_vec([nao, nao], dm.data.clone()).unwrap();
    // T = D @ A_0  ([nao,nao] x [nao,ng]); BLAS-3 instead of a scalar triple loop.
    let a0m = MatrixFull::from_vec([nao, ng], ao[0..nao * ng].to_vec()).unwrap();
    let mut tm = MatrixFull::new([nao, ng], 0.0);
    _dgemm_full(&dm_mat, 'N', &a0m, 'N', &mut tm, 1.0, 0.0);
    let t = &tm.data;
    let a0 = &ao[0..nao * ng];
    let mut rho = vec![0.0; ng * nvar];
    for g in 0..ng {
        let base = g * nao;
        let mut r = 0.0;
        for mu in 0..nao {
            let idx = base + mu;
            r += a0[idx] * t[idx];
        }
        rho[g] = scale * r;
    }
    if nvar == 4 {
        for d in 0..3 {
            let ad = &ao[(1 + d) * nao * ng..(2 + d) * nao * ng];
            for g in 0..ng {
                let base = g * nao;
                let mut r = 0.0;
                for mu in 0..nao {
                    let idx = base + mu;
                    r += ad[idx] * t[idx];
                }
                rho[g + (1 + d) * ng] = 2.0 * scale * r;
            }
        }
    }
    rho
}

fn contract_fxc(rho1: &[f64], fxc: &[f64], weights: &[f64], ng: usize, nvar: usize) -> Vec<f64> {
    // fxc layout [ng,nvar,nvar] col-major: (g,x,y) = g + x*ng + y*nvar*ng
    let mut wv = vec![0.0; nvar * ng];
    for g in 0..ng {
        let w = weights[g];
        for x in 0..nvar {
            let mut s = 0.0;
            for y in 0..nvar {
                s += rho1[y * ng + g] * fxc[g + x * ng + y * nvar * ng];
            }
            wv[x * ng + g] = s * w;
        }
    }
    wv
}

/// Reconstruct the KXC contraction
/// `wv[x] = sum_{y,z} rho1_y rho1_z kxc[x,y,z]` by central-differencing the
/// second derivative `fxc` along the response direction `rho1`.
///
/// This is only used when libxc was built with `XC_DONT_COMPILE_KXC` (i.e.
/// `kxc` is unavailable) and reproduces, to finite-difference accuracy, the
/// analytic contraction PySCF obtains from `ni.eval_xc_eff(..., deriv=3)`.
/// The perturbation is applied to the *generalized* density `(rho, grad rho)`
/// so that `sigma = |grad rho|^2` is perturbed consistently.
fn contract_kxc_fd(
    func_ids: &Vec<usize>,
    func_factors: &Vec<f64>,
    xc_type: XCType,
    rho: &[f64],
    rho1: &[f64],
    weights: &[f64],
    ng: usize,
    nvar: usize,
    eps: f64,
) -> Vec<f64> {
    // When the response direction is identically zero the contraction is zero;
    // skip the two extra libxc evaluations entirely (this is the case in the
    // zero-amplitude assembly, which would otherwise pay for the whole FD).
    if rho1.iter().all(|&v| v == 0.0) {
        return vec![0.0; nvar * ng];
    }
    let mut rho_p = rho.to_vec();
    let mut rho_m = rho.to_vec();
    let mut steps = vec![0.0; ng];
    for g in 0..ng {
        let m1 = (0..nvar).map(|v| rho1[v * ng + g].abs()).fold(0.0, f64::max);
        // scale with the *local* generalized density (not floored at 1, which
        // would grossly over-perturb the low-density tail)
        let m0 = (0..nvar).map(|v| rho[v * ng + g].abs()).fold(0.0, f64::max).max(1.0e-12);
        let h = eps * m0 / m1.max(1.0e-30);
        steps[g] = h;
        for v in 0..nvar {
            let d = h * rho1[v * ng + g];
            rho_p[v * ng + g] = rho[v * ng + g] + d;
            rho_m[v * ng + g] = rho[v * ng + g] - d;
        }
    }
    let fxc_of = |r: &[f64]| -> Vec<f64> {
        let tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, r, ng, 2);
        tensor_slice(tensors[2].as_ref().expect("fxc"), ng * nvar * nvar)
    };
    let fp = fxc_of(&rho_p);
    let fm = fxc_of(&rho_m);
    let mut wv = vec![0.0; nvar * ng];
    for g in 0..ng {
        let w = weights[g];
        let inv2h = 1.0 / (2.0 * steps[g]);
        for x in 0..nvar {
            let mut s = 0.0;
            for b in 0..nvar {
                let idx = g + x * ng + b * nvar * ng;
                s += rho1[b * ng + g] * (fp[idx] - fm[idx]) * inv2h;
            }
            wv[x * ng + g] = s * w;
        }
    }
    wv
}

fn contract_kxc(rho1: &[f64], kxc: &[f64], weights: &[f64], ng: usize, nvar: usize) -> Vec<f64> {
    // kxc layout [ng,nvar,nvar,nvar] col-major
    let mut wv = vec![0.0; nvar * ng];
    for g in 0..ng {
        let w = weights[g];
        for x in 0..nvar {
            let mut s = 0.0;
            for y in 0..nvar {
                for z in 0..nvar {
                    s += rho1[y * ng + g] * rho1[z * ng + g]
                        * kxc[g + x * ng + y * nvar * ng + z * nvar * nvar * ng];
                }
            }
            wv[x * ng + g] = s * w;
        }
    }
    wv
}

fn scale_weighted(v: &[f64], weights: &[f64], ng: usize, nvar: usize) -> Vec<f64> {
    // v layout [ng,nvar] col-major: (g,x) = g + x*ng
    let mut wv = vec![0.0; nvar * ng];
    for g in 0..ng {
        for x in 0..nvar {
            wv[x * ng + g] = v[g + x * ng] * weights[g];
        }
    }
    wv
}

/// PySCF `_lda_eval_mat_` / `_gga_eval_mat_` contraction into `vmat [4,nao,nao]`.
fn xc_eval_mat(
    vmat: &mut [f64],
    ao: &[f64],
    nao: usize,
    ng: usize,
    ao_comp: usize,
    xc_type: XCType,
    wv: &[f64],
    ao_ten: Option<TsrView<'_, f64>>,
) {
    match xc_type {
        XCType::LDA => {
            // vmat[k] += A_k @ (diag(wv) A_0)^T   (BLAS-3 instead of a
            // O(nao^2 * ng) scalar triple loop)
            let mut aow0 = vec![0.0; nao * ng];
            for g in 0..ng {
                for nu in 0..nao {
                    aow0[nu + g * nao] = ao[nu + g * nao] * wv[g];
                }
            }
            let aowm = MatrixFull::from_vec([nao, ng], aow0).unwrap();
            for k in 0..4 {
                let akm =
                    MatrixFull::from_vec([nao, ng], ao[k * nao * ng..(k + 1) * nao * ng].to_vec())
                        .unwrap();
                let mut tmp = MatrixFull::new([nao, nao], 0.0);
                _dgemm_full(&akm, 'N', &aowm, 'T', &mut tmp, 1.0, 0.0);
                for nu in 0..nao {
                    for mu in 0..nao {
                        vmat[k * nao * nao + mu + nu * nao] += tmp[[mu, nu]];
                    }
                }
            }
        }
        XCType::GGA => {
            let mut wvv = wv.to_vec();
            for g in 0..ng {
                wvv[g] *= 0.5;
            }
            // tmp[mu,nu] = sum_g ao[0,mu,g] * aow[nu,g], aow[nu,g]=sum_c ao[c,nu,g] wv[c,g]
            let mut aow_all = vec![0.0; nao * ng];
            for g in 0..ng {
                for nu in 0..nao {
                    let mut aow = 0.0;
                    for c in 0..4 {
                        aow += ao_val(ao, nao, ng, nu, g, c) * wvv[c * ng + g];
                    }
                    aow_all[nu + g * nao] = aow;
                }
            }
            // tmp = A_0 @ aow^T  ([nao,nao]); add tmp + tmp^T (BLAS-3).
            let a0m = MatrixFull::from_vec([nao, ng], ao[0..nao * ng].to_vec()).unwrap();
            let aowm = MatrixFull::from_vec([nao, ng], aow_all).unwrap();
            let mut tmp = MatrixFull::new([nao, nao], 0.0);
            _dgemm_full(&a0m, 'N', &aowm, 'T', &mut tmp, 1.0, 0.0);
            for nu in 0..nao {
                for mu in 0..nao {
                    let t = tmp[[mu, nu]];
                    vmat[mu + nu * nao] += t;
                    vmat[nu + mu * nao] += t;
                }
            }
            // derivative tables via REST's gga_grad_sum.  The AO tensor is
            // built once per grid block by the caller: copying the whole
            // (nao, ng, ao_comp) block on every call was ~1 GB of memcpy per
            // call and dominated this routine.
            let device = DeviceBLAS::default();
            let wv_ten: Tsr<f64> = rt::asarray((wvv.clone(), [ng, 4].f(), &device));
            let mut v: Tsr<f64> = rt::zeros(([nao, nao, 3].f(), &device));
            let ao_view = ao_ten.expect("xc_eval_mat: GGA requires a prebuilt AO tensor");
            crate::grad::rks::gga_grad_sum(&mut v, ao_view, wv_ten.view());
            let vd = tensor_slice(&v, nao * nao * 3);
            for t in 0..3 {
                for nu in 0..nao {
                    for mu in 0..nao {
                        vmat[(1 + t) * nao * nao + mu + nu * nao] += vd[mu + nu * nao + t * nao * nao];
                    }
                }
            }
        }
        _ => panic!("TDDFT gradient: only LDA and GGA supported"),
    }
}

// ---- dense helpers ----

fn build_dm(cl: &[f64], m: &[f64], ncl: usize, cr: &[f64], ncr: usize, nao: usize) -> AOMat {
    let lm = MatrixFull::from_vec([nao, ncl], cl.to_vec()).unwrap();
    let mm = MatrixFull::from_vec([ncl, ncr], m.to_vec()).unwrap();
    let rm = MatrixFull::from_vec([nao, ncr], cr.to_vec()).unwrap();
    let mut t = MatrixFull::new([nao, ncr], 0.0);
    _dgemm_full(&lm, 'N', &mm, 'N', &mut t, 1.0, 0.0);
    let mut out = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t, 'N', &rm, 'T', &mut out, 1.0, 0.0);
    AOMat::from_matrixfull(&out)
}

/// `l = cl @ m` as a `[nao, ncr]` col-major block — the left factor of the
/// density `cl m cr^T`.  Lets a factor be built from an already-computed left
/// block (used by the `U_pm = C_v (X +/- Y)` representation, where left and
/// right factors are the same matrix).
fn left_factor(cl: &[f64], m: &[f64], ncl: usize, ncr: usize, nao: usize) -> Vec<f64> {
    let lm = MatrixFull::from_vec([nao, ncl], cl.to_vec()).unwrap();
    let mm = MatrixFull::from_vec([ncl, ncr], m.to_vec()).unwrap();
    let mut t = MatrixFull::new([nao, ncr], 0.0);
    _dgemm_full(&lm, 'N', &mm, 'N', &mut t, 1.0, 0.0);
    t.data
}

/// `D = l r^T` -> `D^T = r l^T`, with the column count unchanged.  Rebuilding
/// the transpose by re-factorising (`C_o X^T C_v^T`) instead inflates the rank
/// from `nocc` to `nvir`.
fn transpose_rank(f: &RankFactor) -> RankFactor {
    RankFactor { l: f.r.clone(), r: f.l.clone(), nfac: f.nfac }
}

fn rank_factor_product(cl: &[f64], m: &[f64], ncl: usize, cr: &[f64], ncr: usize, nao: usize) -> RankFactor {
    // D = cl m cr^T = (cl m) cr^T
    let lm = MatrixFull::from_vec([nao, ncl], cl.to_vec()).unwrap();
    let mm = MatrixFull::from_vec([ncl, ncr], m.to_vec()).unwrap();
    let mut t = MatrixFull::new([nao, ncr], 0.0);
    _dgemm_full(&lm, 'N', &mm, 'N', &mut t, 1.0, 0.0);
    RankFactor { l: t.data, r: cr.to_vec(), nfac: ncr }
}



/// `D -> s*D` for a rank-factor representation `D = sum_i l_i r_i^T`.
fn scale_rank(f: &RankFactor, s: f64) -> RankFactor {
    RankFactor { l: f.l.iter().map(|v| v * s).collect(), r: f.r.clone(), nfac: f.nfac }
}

fn neg_rank(f: &RankFactor) -> RankFactor {
    RankFactor { l: f.l.clone(), r: f.r.iter().map(|v| -v).collect(), nfac: f.nfac }
}

fn transpose_sq(m: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut o = vec![0.0; r * c];
    for j in 0..c {
        for i in 0..r {
            o[j + i * c] = m[i + j * r];
        }
    }
    o
}

fn transpose(a: &AOMat, nao: usize) -> AOMat {
    let mut o = AOMat::zeros(nao);
    for nu in 0..nao {
        for mu in 0..nao {
            o.set(mu, nu, a.get(nu, mu));
        }
    }
    o
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i + i * n] = 1.0;
    }
    m
}

/// `out[i + a*nocc] = C_vir[:,a]^T V C_occ[:,i]`.
///
/// Computed as `C_occ^T (V^T C_vir)` — two BLAS-3 GEMMs, `O(nao^2*nvir +
/// nao*nocc*nvir)`, replacing the four-deep scalar loop that was `O(nao^2*`
/// `nocc*nvir)`.  The product is formed as an `[nocc,nvir]` matrix whose
/// column-major layout *is* the required `i + a*nocc` layout, so no extra
/// transpose is needed.
fn project_vo(v: &AOMat, c_occ: &[f64], c_vir: &[f64], nao: usize, nocc: usize, nvir: usize) -> Vec<f64> {
    let vm = MatrixFull::from_vec([nao, nao], v.data.clone()).unwrap();
    let cv = MatrixFull::from_vec([nao, nvir], c_vir.to_vec()).unwrap();
    let co = MatrixFull::from_vec([nao, nocc], c_occ.to_vec()).unwrap();
    let mut t = MatrixFull::new([nao, nvir], 0.0);
    _dgemm_full(&vm, 'T', &cv, 'N', &mut t, 1.0, 0.0);
    let mut out = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(&co, 'T', &t, 'N', &mut out, 1.0, 0.0);
    out.data
}

/// `out[i + j*nocc] = C_occ[:,i]^T V C_occ[:,j]`, two BLAS-3 GEMMs.
fn project_oo(v: &AOMat, c_occ: &[f64], nao: usize, nocc: usize) -> Vec<f64> {
    let vm = MatrixFull::from_vec([nao, nao], v.data.clone()).unwrap();
    let co = MatrixFull::from_vec([nao, nocc], c_occ.to_vec()).unwrap();
    let mut t = MatrixFull::new([nao, nocc], 0.0);
    _dgemm_full(&vm, 'N', &co, 'N', &mut t, 1.0, 0.0);
    let mut out = MatrixFull::new([nocc, nocc], 0.0);
    _dgemm_full(&co, 'T', &t, 'N', &mut out, 1.0, 0.0);
    out.data
}

fn project_full(v: &AOMat, c: &MatrixFull<f64>, nao: usize, nmo: usize) -> Vec<f64> {
    let vm = MatrixFull::from_vec([nao, nao], v.data.clone()).unwrap();
    let mut tmp = MatrixFull::new([nao, nmo], 0.0);
    _dgemm_full(&vm, 'N', c, 'N', &mut tmp, 1.0, 0.0);
    let mut out = MatrixFull::new([nmo, nmo], 0.0);
    _dgemm_full(c, 'T', &tmp, 'N', &mut out, 1.0, 0.0);
    out.data
}

#[allow(clippy::too_many_arguments)]
fn apply_vo_coupling(
    wvo: &mut [f64],
    mo_mat: &[f64],
    amp: &[f64],
    nocc: usize,
    nvir: usize,
    nmo: usize,
    start_mo: usize,
    lumo: usize,
    f_oo: f64,
    f_vv: f64,
) {
    for a in 0..nvir {
        for i in 0..nocc {
            let idx = i + a * nocc;
            // PySCF: `wvo[a,k] -= 2 sum_i veff0mop[k,i] xpy[a,i]` (see the
            // `einsum('ki,ai->ak', ...)` calls in tdrks.grad_elec).  The output
            // index of REST's `idx = i + a*nocc` is the *second* MO index, so
            // the matrix element must be read as `M[i, j]` (row = output index,
            // column = the summed index).  For `veff0mop` the block is
            // symmetric so this is invisible; `veff0mom` (K of the
            // antisymmetric `dmc`) is not, and the transposed read was wrong
            // for hybrids.
            let mut s = 0.0;
            for j in 0..nocc {
                s += mo_mat[(start_mo + i) + (start_mo + j) * nmo] * amp[a + j * nvir];
            }
            wvo[idx] += f_oo * s;
            let mut s2 = 0.0;
            for b in 0..nvir {
                s2 += mo_mat[(lumo + b) + (lumo + a) * nmo] * amp[b + i * nvir];
            }
            wvo[idx] += f_vv * s2;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transform_mo_to_ao(
    c_occ: &[f64],
    c_vir: &[f64],
    acc: &[f64],
    nmo: usize,
    nocc: usize,
    nvir: usize,
    start_mo: usize,
    lumo: usize,
    nao: usize,
) -> AOMat {
    let mut c = vec![0.0; nao * nmo];
    for i in 0..nocc {
        for mu in 0..nao {
            c[mu + (start_mo + i) * nao] = c_occ[mu + i * nao];
        }
    }
    for a in 0..nvir {
        for mu in 0..nao {
            c[mu + (lumo + a) * nao] = c_vir[mu + a * nao];
        }
    }
    let cm = MatrixFull::from_vec([nao, nmo], c).unwrap();
    let am = MatrixFull::from_vec([nmo, nmo], acc.to_vec()).unwrap();
    let mut t = MatrixFull::new([nao, nmo], 0.0);
    _dgemm_full(&cm, 'N', &am, 'N', &mut t, 1.0, 0.0);
    let mut out = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t, 'N', &cm, 'T', &mut out, 1.0, 0.0);
    AOMat::from_matrixfull(&out)
}

