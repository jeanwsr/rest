/// Frequency-independent data of one orbital row's self-energy.
///
/// `m = s_metric * q_n` and the imaginary-axis table
/// `w_imag[wi][m] = W_c[n, m](iu_wi)` do **not** depend on the trial QP energy,
/// so the Newton iterations of one orbital share them.  Building the table is
/// the dominant cost of a self-energy evaluation (`nquad` projections against
/// the low-rank factors) and used to be repeated for every trial `omega` — a
/// factor equal to the iteration count.
pub struct QpRowTable {
    /// `m = s_metric * q_n`, `[naux, nmo]` col-major
    pub m: Vec<f64>,
    /// `W_c[n, m](iu_wi)` per quadrature point
    pub w_imag: Vec<Vec<f64>>,
    /// Static row `W_c[n, m](0)` (empty when the engine runs raw/full-CD).
    /// Shared by every trial frequency of the Newton loop, so it is computed
    /// once with the table instead of once per evaluation.
    pub w0: Vec<f64>,
}

impl<'a> GwGradEngine<'a> {
    pub fn new(scf: &'a SCF, config: GwGradConfig) -> Self {
        let raw = build_raw_ri_tensors(scf);
        Self::new_with_raw(scf, config, raw)
    }

    /// Engine with the grid-derived `v_xc` supplied by the caller.
    ///
    /// Use this together with [`Self::precompute_vxc`] when the caller owns the `SCF`
    /// mutably and wants to release the SCF grid's tabulated AO tables in between: those
    /// tables are read *only* by `precompute_vxc` in the whole GW/BSE analytic-gradient
    /// path, so they are dead weight from the moment it returns.
    pub fn new_with_vxc(scf: &'a SCF, config: GwGradConfig, vxc: GwVxc) -> Self {
        let raw = build_raw_ri_tensors(scf);
        Self::new_with_raw_and_vxc(scf, config, raw, vxc)
    }

    /// The two grid-derived `v_xc` objects this engine needs, computed up front.
    ///
    /// This is the **only** consumer of the SCF grid's tabulated AO tables
    /// (`grids.ao`/`aop`/`ao_compressed`/`aop_compressed`/`non0tab`) in the GW/BSE analytic
    /// gradient: `ri_gw`/`ri_bse` never touch `scf.grids`, and every other DFT piece of the
    /// gradient — the CP-KS response (`NIMatmul`), the f_xc kernel (`FxcHessianCache`) and
    /// the explicit `d v_xc/dR` (`vxc_deriv1_streaming`) — evaluates its own AO from
    /// `grids.coordinates`/`weights`.  Hence a driver holding the `SCF` mutably can compute
    /// these, call `Grids::release_tabulated_ao` and only then build the engine.
    pub fn precompute_vxc(scf: &SCF) -> GwVxc {
        // exactly the dimensions `new_with_raw_and_vxc` uses (`raw.nao` == `mol.num_basis`,
        // `nmo` == `mol.num_state`); `c_mo` may carry more columns than `nmo`, and only the
        // leading `nmo` of them are contracted.
        let nao = scf.mol.num_basis;
        let nmo = scf.mol.num_state;
        let c_mo = &scf.eigenvectors[0];
        let nn = crate::ri_gw::vxc_ao2mo(scf);
        let mo: Vec<f64> = if !scf.mol.dfadef.is_dfa_scf() {
            Vec::new()
        } else {
            let vxc_ao = scf
                .generate_vxc_rayon(1.0)
                .2[0]
                .to_matrixfull()
                .expect("gw_grad: v_xc AO matrix")
                .data;
            let t = gemm_nt(&c_mo.data, nao, nmo, &vxc_ao, nao, nao);
            gemm_nn(&t, nmo, nao, &c_mo.data, nao, nmo)
        };
        GwVxc { nn, mo }
    }

    /// Engine with externally supplied raw RI tensors (allows perturbing the
    /// integrals directly, e.g. for validation).
    pub fn new_with_raw(scf: &'a SCF, config: GwGradConfig, raw: RawRiTensors) -> Self {
        let vxc = Self::precompute_vxc(scf);
        Self::new_with_raw_and_vxc(scf, config, raw, vxc)
    }

    /// As [`Self::new_with_raw`], with the grid-derived `v_xc` supplied by the caller.
    pub fn new_with_raw_and_vxc(scf: &'a SCF, mut config: GwGradConfig, raw: RawRiTensors, vxc: GwVxc) -> Self {
        assert_eq!(scf.mol.spin_channel, 1, "gw_grad: restricted closed-shell only");
        assert_eq!(scf.mol.start_mo, 0, "gw_grad: no frozen orbitals supported");
        assert!(
            scf.mol.dfadef.omega().is_none(),
            "gw_grad: range-separated hybrid references are not supported yet"
        );
        if config.num_freq == 0 {
            config.num_freq = 20;
        }
        let nao = raw.nao;
        let naux = raw.naux;
        let nmo = scf.mol.num_state;
        let nocc = scf.homo[0] + 1;
        let natm = raw.natm;
        let e = scf.eigenvalues[0][..nmo].to_vec();
        let c_mo = scf.eigenvectors[0].clone();

        // ---- raw q[P, p, q] = sum_mu nu I[mu nu P] C[mu p] C[nu q] ----
        // Two dgemms per auxiliary function (`dI_P C` then `C^T (dI_P C)`)
        // instead of the naive O(nao^2 nmo) triple loop: this is the dominant
        // setup cost for large basis sets (BLAS runs it at cache speed).
        let mut q = vec![0.0f64; naux * nmo * nmo];
        {
            // The three-centre integrals are consumed **auxiliary-atom block by block**: the dense
            // `nao^2 * naux` tensor (874 MB at naphthalene/def2-TZVP) is never materialised, which
            // removes it from the engine-build peak. The integral work is the same shell-triplet
            // sum as one `int3c2e` call.
            let n2 = nao * nao;
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            raw.for_each_i3_atom_block(|p0, nb, blk| {
                for pl in 0..nb {
                    let p_aux = p0 + pl;
                    ip.data.copy_from_slice(&blk[pl * n2..(pl + 1) * n2]);
                    // tmp[mu, q_mo] = sum_nu I[mu, nu, P] C[nu, q_mo]
                    _dgemm_full(&ip, 'N', &c_mo, 'N', &mut tmp, 1.0, 0.0);
                    // sl[p_mo, q_mo] = sum_mu C[mu, p_mo] tmp[mu, q_mo]
                    _dgemm_full(&c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                    for col in 0..nmo * nmo {
                        q[col * naux + p_aux] = sl.data[col];
                    }
                }
            });
        }
        symmetrise_pairs(&mut q, naux, nmo);
        // pair-slowest repack (strided once, then contiguous for the gemms).
        // Only the legacy `qx_from_u_blocks` route reads it (`U^T q + q U` as one
        // gemm); that route repacks per auxiliary block on the fly inside
        // `qx_from_u` when this is empty. The folded route (`folded_qx`) never
        // touches `q_pairmajor`, so it is not built there — that keeps
        // `naux * nmo^2` doubles (874 MB at naphthalene/def2-TZVP) out of the
        // resident set of every optimised gradient. Same for the full-CD
        // engine (`skip_lr_cache`).
        let q_pairmajor: Vec<f64> = if config.skip_lr_cache || config.folded_qx {
            Vec::new()
        } else {
            let npair = nmo * nmo;
            let mut v = vec![0.0f64; naux * npair];
            for pair in 0..npair {
                for p in 0..naux {
                    v[pair + p * npair] = q[pair * naux + p];
                }
            }
            v
        };

        mem_probe("  build: after q");
        // ---- metric and J^{-1/2} ----
        let j_mat = MatrixFull::from_vec([naux, naux], raw.j2.clone()).unwrap();
        let mut j_chol = j_mat.clone();
        _dpotrf(&mut j_chol, 'L');
        let (vals, vecs) = eigh_symmetric(&j_mat.data, naux);
        // `S = J^{-1/2} = V diag(1/sqrt(lam)) V^T` as two BLAS-3 products: the
        // explicit `O(naux^3)` scalar triple loop this replaces (with its
        // branch on the eigenvalue cutoff) cost ≈ 1.2 s at naphthalene/def2-SVP.
        let mut vd = vec![0.0f64; naux * naux];
        for v in 0..naux {
            let scale = if vals[v] > 1.0e-10 { 1.0 / vals[v].sqrt() } else { 0.0 };
            for a in 0..naux {
                vd[a + v * naux] = vecs[a + v * naux] * scale;
            }
        }
        let mut sm = MatrixFull::new([naux, naux], 0.0);
        {
            let vd_m = to_mat(&vd, naux, naux);
            let v_m = to_mat(&vecs, naux, naux);
            _dgemm_full(&vd_m, 'N', &v_m, 'T', &mut sm, 1.0, 0.0);
        }
        // keep `S` exactly symmetric: the self-energy value identities
        // (`sum_P m corr`, with `m = S q_n`) rely on it
        let mut s_metric = sm.data;
        for a in 0..naux {
            for b in (a + 1)..naux {
                let avg = 0.5 * (s_metric[a + b * naux] + s_metric[b + a * naux]);
                s_metric[a + b * naux] = avg;
                s_metric[b + a * naux] = avg;
            }
        }

        // ---- metric-transformed integrals (LR-only), occupied MO columns only ----
        // `consts(n)` reads `qt[P, n, i]` for occupied `i` only, i.e. the pair columns
        // `pair_col(n, i) = n + i*nmo < nocc*nmo` — the first `nocc*nmo` columns of `q`; and `qtia`
        // is the (`n >= nocc`, `i < nocc`) block, which (because `q` is pair-symmetrised, so
        // `q[(i,a)] == q[(a,i)]` exactly) is the (`n = a`, `i < nocc`) block of that same prefix.
        // Keeping only the prefix stores `naux*nmo*nocc` doubles (87 MB at naphthalene/def2-TZVP)
        // instead of `naux*nmo^2` (874 MB), which is what used to dominate the engine build.
        let ncol_occ = nocc * nmo;
        let qt = if config.skip_lr_cache {
            Vec::new()
        } else {
            metric_transform_blocked(&s_metric, &q[..ncol_occ * naux], naux, ncol_occ)
        };

        mem_probe("  build: after qt");
        // ---- occupied-virtual blocks ----
        let nov = nocc * (nmo - nocc);
        let mut qia = vec![0.0f64; naux * nov];
        let mut qtia = vec![0.0f64; if config.skip_lr_cache { 0 } else { naux * nov }];
        for a in nocc..nmo {
            for i in 0..nocc {
                let src = (i + a * nmo) * naux;
                let dst = (i + (a - nocc) * nocc) * naux;
                qia[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
                if !config.skip_lr_cache {
                    // occupied-column prefix index of the same number (`q` is pair-symmetric)
                    let src_occ = (a + i * nmo) * naux;
                    qtia[dst..dst + naux].copy_from_slice(&qt[src_occ..src_occ + naux]);
                }
            }
        }
        let mut de_ia: Vec<f64> = Vec::with_capacity(nov);
        for a in nocc..nmo {
            for i in 0..nocc {
                de_ia.push(e[a] - e[i]);
            }
        }
        assert_eq!(de_ia.len(), nov);

        mem_probe("  build: after qia/qtia");
        // ---- imaginary-axis quadrature (REST grids) ----
        let quad: Vec<(f64, f64)> = match config.freq_grid_type {
            1 => {
                let (om, w) =
                    gauss_legendre_grids([0.0, config.freq_cut_off], config.num_freq);
                om.into_iter().zip(w).collect()
            }
            2 => {
                let (om, w) = logarithmic_grid([0.0, config.freq_cut_off], config.num_freq);
                om.into_iter().zip(w).collect()
            }
            _ => {
                let (om, w) = trans_gauss_legendre_grids(1.0, config.num_freq);
                om.into_iter().zip(w).collect()
            }
        };

        // ---- REST real-axis grid (de_max scan + linear/quadratic nodes;
        //      LR-only) ----
        let (z_grid, z_grid_sens, z_grid_mover) = if config.skip_lr_cache {
            (Vec::new(), Vec::new(), None)
        } else {
            let nsemin = (nocc - 1).saturating_sub(config.selfenergy_state_range);
            let nsemax = (nocc + config.selfenergy_state_range).min(nmo - 1);
            let g = build_real_axis_grid(
                &e,
                nocc,
                config.nomega_chi_real,
                nsemin,
                nsemax,
                config.nomega_sigma,
                config.step_sigma,
                config.res_tol,
                config.grid_type,
                config.omega_chi_max,
                config.degenerate_tol,
            );
            // `REST_GWGRAD_ZGRID=1` dumps the real-axis layout: the node count,
            // the range and the spacing decide how well the quantised residue
            // protocol resolves a pole, and the auto rule rarely does what the
            // `nomega_chi_real` value suggests.
            if std::env::var("REST_GWGRAD_ZGRID").is_ok() {
                eprintln!(
                    "[zgrid] n={} range={:.6}..{:.6} d={:.6}",
                    g.nodes.len(),
                    g.nodes.first().copied().unwrap_or(0.0),
                    g.nodes.last().copied().unwrap_or(0.0),
                    g.nodes.windows(2).next().map(|w| w[1] - w[0]).unwrap_or(0.0)
                );
            }
            (g.nodes, g.node_sens, g.mover)
        };

        // ---- reference static-term data and low-rank factors (LR-only) ----
        let hybrid = scf.mol.dfadef.dfa_hybrid_scf;
        // a plain HF reference counts as a hybrid with alpha = 1: its orbital
        // energies already carry the exact exchange (see `ri_gw::exchange_fraction`)
        let fock_hyb = crate::ri_gw::exchange_fraction(scf);
        // supplied by `precompute_vxc`: the caller may have released the SCF grid's
        // tabulated AO tables already (see `Self::precompute_vxc`)
        let GwVxc { nn: vxc_nn, mo: vxc_mo } = vxc;
        let (shells, shell_of) = degenerate_shells(&e, config.degenerate_tol);
        if config.degenerate_tol > 0.0 && std::env::var("REST_GWGRAD_DBG").is_ok() {
            let deg: Vec<&Vec<usize>> = shells.iter().filter(|s| s.len() > 1).collect();
            println!(
                "[new DBG] degenerate shells (tol {:.1e}): {:?}",
                config.degenerate_tol, deg
            );
        }
        if config.skip_lr_cache {
            GwGradEngine {
                scf,
                config,
                raw,
                nao,
                naux,
                nmo,
                nocc,
                natm,
                e,
                pinned_qp: Vec::new(),
                c_mo,
                q: RefCell::new(q),
                q_pairmajor,
                j_mat,
                j_chol,
                s_metric,
                qt: RefCell::new(qt),
                qia,
                de_ia,
                quad,
                z_grid: z_grid.clone(),
                z_grid_sens,
                z_grid_mover,
                lr_imag: RefCell::new(Vec::new()),
                lr_grid: (0..z_grid.len()).map(|_| OnceLock::new()).collect(),
                lr_static: RefCell::new(None),
                qtia: RefCell::new(qtia),
                wac_ref: RefCell::new(None),
                hybrid,
                fock_hyb,
                vxc_nn,
                vxc_mo,
                shells,
                shell_of,
            }
        } else {
            if std::env::var("REST_GWGRAD_DBG").is_ok() {
                println!("[new DBG] qtia max = {}", qtia.iter().fold(0.0f64, |a, &x| a.max(x.abs())));
            }
            // The imaginary-axis factors are needed by every self-energy
            // evaluation; the real-axis grid factors are built on demand
            // (`ensure_lr_grid_factor`), because a self-energy row reads at
            // most a handful of grid points while a full grid costs one
            // `naux x naux` eigendecomposition per point.
            let lr_imag: Vec<LrFactor> = quad
                .iter()
                .map(|&(u, _)| {
                    build_lr_factor(
                        &qtia, naux, nov, &de_ia, u, 'I', 0.0, config.low_rank_tolerance,
                    )
                })
                .collect();
            let lr_grid: Vec<Option<LrFactor>> = (0..z_grid.len()).map(|_| None).collect();
            mem_probe("  build: after all factors");
            if std::env::var("REST_GWGRAD_DBG").is_ok() {
                println!("[new DBG] lr_imag nkeeps = {:?}", lr_imag.iter().map(|f| f.eigval.len()).collect::<Vec<_>>());
            }
            GwGradEngine {
                scf,
                config,
                raw,
                nao,
                naux,
                nmo,
                nocc,
                natm,
                e,
                pinned_qp: Vec::new(),
                c_mo,
                q: RefCell::new(q),
                q_pairmajor,
                j_mat,
                j_chol,
                s_metric,
                qt: RefCell::new(qt),
                qia,
                de_ia,
                quad,
                z_grid,
                z_grid_sens,
                z_grid_mover,
                lr_imag: RefCell::new(lr_imag),
                lr_grid: lr_grid
                    .into_iter()
                    .map(|f| {
                        let cell = OnceLock::new();
                        if let Some(v) = f {
                            let _ = cell.set(v);
                        }
                        cell
                    })
                    .collect(),
                lr_static: RefCell::new(None),
                qtia: RefCell::new(qtia),
                wac_ref: RefCell::new(None),
                hybrid,
                fock_hyb,
                vxc_nn,
                vxc_mo,
                shells,
                shell_of,
            }
        }
    }

    fn pair_col(&self, p: usize, q: usize) -> usize {
        p + q * self.nmo
    }

    /// True when `p` and `q` belong to the same (size > 1) degenerate shell.
    pub fn same_degenerate_shell(&self, p: usize, q: usize) -> bool {
        p != q && self.shell_of[p] == self.shell_of[q] && self.shells[self.shell_of[p]].len() > 1
    }

    /// The symmetry-protected degenerate shell containing `target`
    /// (`vec![target]` when the orbital is non-degenerate).
    pub fn degenerate_shell(&self, target: usize) -> Vec<usize> {
        self.shells[self.shell_of[target]].clone()
    }

    /// Gather the block of one MO index against all MOs: `[naux, nmo]` with
    /// columns `src[:, n, m]`.
    fn gather_row(&self, src: &[f64], n: usize) -> Vec<f64> {
        let naux = self.naux;
        let mut out = vec![0.0f64; naux * self.nmo];
        for m in 0..self.nmo {
            let base = self.pair_col(n, m) * naux;
            out[m * naux..(m + 1) * naux].copy_from_slice(&src[base..base + naux]);
        }
        out
    }

    /// Gather one raw pair column `src[:, n, m]`.
    fn gather_pair(&self, src: &[f64], n: usize, m: usize) -> Vec<f64> {
        let base = self.pair_col(n, m) * self.naux;
        src[base..base + self.naux].to_vec()
    }

    /// `y0 = J^{-1} rhs` for n_col right-hand sides of length naux.
    fn metric_solve(&self, rhs: &[f64], n_col: usize) -> Vec<f64> {
        let qt = gemm_nn(&self.s_metric, self.naux, self.naux, rhs, self.naux, n_col);
        gemm_nn(&self.s_metric, self.naux, self.naux, &qt, self.naux, n_col)
    }

    /// `J^{-1} B` through the pre-computed Cholesky factor, with
    /// `B [naux, ncol]` col-major, overwritten with the solution
    /// (multi-right-hand-side triangular solves, one factorisation per
    /// geometry).
    pub(crate) fn solve_metric_batch(&self, b: &mut [f64], ncol: usize) {
        let naux = self.naux;
        let mut bm = to_mat(b, naux, ncol);
        let ok1 = _dtrtrs(&self.j_chol, &mut bm, 'L', 'N', 'N');
        let ok2 = _dtrtrs(&self.j_chol, &mut bm, 'L', 'T', 'N');
        assert!(ok1 && ok2, "gw_grad: metric triangular solve failed");
        b.copy_from_slice(&bm.data);
    }

    /// `J^{-1} rhs` for a single right-hand side.
    pub(crate) fn solve_metric(&self, rhs: &[f64]) -> Vec<f64> {
        let mut b = rhs.to_vec();
        self.solve_metric_batch(&mut b, 1);
        b
    }

    /// Ensure the low-rank factors are resident (rebuilds them from `qtia` after `evict_qt`).
    fn ensure_lr(&self) {
        // Only the imaginary-axis table lives here; the real-axis grid is one
        // `OnceLock` per node (see the field), so this function no longer owns it
        // and cannot be re-entered from `ensure_lr_grid_factor`.
        let need = {
            let g = self.lr_imag.borrow();
            g.is_empty()
        };
        if need && !self.config.skip_lr_cache {
            self.with_qtia(|qtia| {
                let imag: Vec<LrFactor> = self
                    .quad
                    .iter()
                    .map(|&(u, _)| {
                        build_lr_factor(
                            qtia,
                            self.naux,
                            self.nocc * (self.nmo - self.nocc),
                            &self.de_ia,
                            u,
                            'I',
                            0.0,
                            self.config.low_rank_tolerance,
                        )
                    })
                    .collect();
                *self.lr_imag.borrow_mut() = imag;
            });
        }
    }

    /// Make the real-axis factor of grid point `idx` resident.
    ///
    /// Each slot is its own [`OnceLock`], so several threads may race here: one
    /// builds the factor, the others block until it is published.  That is what
    /// lets a parallel QP-cache sweep fill grid nodes lazily instead of
    /// pre-materialising the whole grid.
    fn ensure_lr_grid_factor(&self, idx: usize) {
        if self.lr_grid[idx].get().is_some() {
            return;
        }
        // A worker may reach this before `warm_lazy_caches` (or after an
        // `evict_lr`); make the low-rank machinery itself resident first.
        self.ensure_lr();
        let z = self.z_grid[idx];
        let factor = self.with_qtia(|qtia| {
            build_lr_factor_at(
                qtia,
                self.naux,
                self.nocc * (self.nmo - self.nocc),
                &self.de_ia,
                z,
                self.config.eta,
                self.config.low_rank_tolerance,
            )
        });
        let _ = self.lr_grid[idx].set(factor);
    }

    /// Make the static (`z = 0`) low-rank factor resident, building it on
    /// first use (see [`Self::lr_static`]).  The `z = 0` Lorentzian diagonal
    /// `-4 de/(de^2 + eta^2)` is exactly what the production real-axis grid
    /// evaluates at its first node, so the subtracted functional stays
    /// consistent with REST's own response protocol.
    fn ensure_lr_static(&self) {
        let have = {
            let g = self.lr_static.borrow();
            g.is_some()
        };
        if have || self.config.skip_lr_cache {
            return;
        }
        let factor = self.with_qtia(|qtia| {
            build_lr_factor(
                qtia,
                self.naux,
                self.nocc * (self.nmo - self.nocc),
                &self.de_ia,
                0.0,
                'R',
                self.config.eta,
                self.config.low_rank_tolerance,
            )
        });
        *self.lr_static.borrow_mut() = Some(factor);
    }

    /// Static response vectors `y0s = (J - Q(0))^-1 q_n` of one orbital row
    /// from its metric intermediate `m = S q_n`.
    fn lr_static_row_response(&self, m: &[f64]) -> Vec<f64> {
        self.ensure_lr_static();
        let guard = self.lr_static.borrow();
        guard
            .as_ref()
            .expect("gw_grad: static low-rank factor missing")
            .response_vectors_from_qt(&self.s_metric, m, self.nmo)
    }

    fn lr_factor_nearest_idx(&self, z: f64) -> usize {
        let zs = &self.z_grid;
        let n = zs.len();
        if n <= 1 || z <= zs[0] {
            return 0;
        }
        if z >= zs[n - 1] {
            return n - 1;
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if zs[mid] < z {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (z - zs[lo]) < (zs[hi] - z) {
            lo
        } else {
            hi
        }
    }

    /// Run `f` on the raw MO integrals `q[P, p, q]` (pair-symmetrised, aux-fastest).
    ///
    /// Rebuilds them from `raw.i3` when a previous task released them with [`Self::evict_q`].
    pub fn with_q<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        self.ensure_q();
        f(&self.q.borrow())
    }

    /// Make sure `q` is resident (see [`Self::with_q`]).
    pub fn ensure_q(&self) {
        if self.q.borrow().is_empty() {
            let built = self.build_q();
            *self.q.borrow_mut() = built;
        }
    }

    /// Rebuild `q[P, p, q] = sum_{mu,nu} (mu nu | P) C[mu,p] C[nu,q]` (pair-symmetrised) from the
    /// raw three-centre integrals; mirrors the construction in the engine's `new_with_raw`.
    fn build_q(&self) -> Vec<f64> {
        let (nao, nmo, naux) = (self.nao, self.nmo, self.naux);
        let n2 = nao * nao;
        let mut q = vec![0.0f64; naux * nmo * nmo];
        // Consume the three-centre integrals **auxiliary-atom block by block**, exactly as the
        // engine build does (`for_each_i3_atom_block`), instead of materialising the dense
        // `nao^2 * naux` tensor through `with_i3`.  This is the *rebuild* path (`ensure_q` after
        // `evict_q`), so it used to be the one place that brought the dense tensor back — 2.8 GB
        // at polyene n=6/def2-TZVPP, held on top of the `q` it is building and alongside the SCF's
        // own `(mu nu|P)`.  Every `q[., P]` is assigned exactly once and in the same auxiliary
        // order as the full-tensor version, so the result is bit-identical; the integral work is
        // the same shell-triplet sum, split into a few more libcint calls.
        {
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            self.raw.for_each_i3_atom_block(|p0, nb, blk| {
                for pl in 0..nb {
                    let p_aux = p0 + pl;
                    ip.data.copy_from_slice(&blk[pl * n2..(pl + 1) * n2]);
                    _dgemm_full(&ip, 'N', &self.c_mo, 'N', &mut tmp, 1.0, 0.0);
                    _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                    for col in 0..nmo * nmo {
                        q[col * naux + p_aux] = sl.data[col];
                    }
                }
            });
        }
        symmetrise_pairs(&mut q, naux, nmo);
        q
    }

    /// Make sure `qt` is resident (see [`Self::with_qt`]).
    fn ensure_qt(&self) {
        let need = {
            let g = self.qt.borrow();
            g.is_empty()
        };
        if need && !self.config.skip_lr_cache {
            self.ensure_q();
            // same occupied-column prefix as the constructor (see `new_with_raw`)
            let ncol_occ = self.nocc * self.nmo;
            let built = metric_transform_blocked(
                &self.s_metric,
                &self.q.borrow()[..ncol_occ * self.naux],
                self.naux,
                ncol_occ,
            );
            *self.qt.borrow_mut() = built;
        }
    }

    /// Run `f` on the metric-transformed integrals `qt` (see [`Self::evict_qt`]).
    pub fn with_qt<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        self.ensure_qt();
        f(&self.qt.borrow())
    }

    /// Run `f` on the occupied-virtual block `qtia` (see [`Self::evict_qt`]).
    pub fn with_qtia<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        let need = {
            let g = self.qtia.borrow();
            g.is_empty()
        };
        if need && !self.config.skip_lr_cache {
            self.ensure_qt();
            let (naux, nmo, nocc) = (self.naux, self.nmo, self.nocc);
            let qt = self.qt.borrow();
            let mut qtia = vec![0.0f64; naux * nocc * (nmo - nocc)];
            for a in nocc..nmo {
                for i in 0..nocc {
                    // occupied-column prefix index of the same number (see `new_with_raw`)
                    let src = (a + i * nmo) * naux;
                    let dst = (i + (a - nocc) * nocc) * naux;
                    qtia[dst..dst + naux].copy_from_slice(&qt[src..src + naux]);
                }
            }
            *self.qtia.borrow_mut() = qtia;
        }
        f(&self.qtia.borrow())
    }

    /// Release `qt` and `qtia` (`naux * nmo^2` + `naux * nov` doubles): they are only read while
    /// building a QP cache (the static `consts` term and the exact-residue on-the-fly factors),
    /// never by a gradient's own contractions. [`Self::with_qt`]/[`Self::with_qtia`] rebuild them.
    pub fn evict_qt(&self) {
        // Every release here is a pure memory optimisation — the `ensure_*`
        // builders rebuild on demand — so a release that is momentarily
        // impossible (a reader still finishing) is skipped rather than turned
        // into a `RefCell already borrowed` panic.  See `evict_qt_blocks`.
        release_vec(&self.qt);
        release_vec(&self.qtia);
        // the low-rank factors are read only while building a QP cache as well
        release_vec(&self.lr_imag);
        if let Ok(mut g) = self.lr_static.try_borrow_mut() {
            *g = None;
        }
    }

    /// Release only the metric-transformed integral blocks (`qt`, `qtia`) and
    /// keep the cached low-rank factors.
    ///
    /// A driver that has finished building its QP caches but still needs the
    /// factors for the gradient pullback (the BSE gradient) can drop
    /// `naux*nmo^2 + naux*nov` doubles here — 76 MB at naphthalene/def2-SVP —
    /// without giving up the factors.  `with_qt`/`with_qtia` rebuild the blocks
    /// from `q` if a later cache build asks for them.
    pub fn evict_qt_blocks(&self) {
        release_vec(&self.qt);
        release_vec(&self.qtia);
    }

    /// Release `q` (`naux * nmo^2` doubles): the folded and adjoint gradient routes read it only
    /// up to the covector fold. [`Self::with_q`] rebuilds it on demand.
    pub fn evict_q(&self) {
        release_vec(&self.q);
    }

    /// REST static term `consts_n = eps_n + (1-alpha) Sigma_x,nn - vxc_nn`
    /// with the RI exchange `Sigma_x,nn = -sum_i (n i | n i)`.
    pub fn consts(&self, n: usize) -> f64 {
        let exchange = self.with_qt(|qt| {
            let mut acc = 0.0;
            for i in 0..self.nocc {
                let base = self.pair_col(n, i) * self.naux;
                acc -= qt[base..base + self.naux].iter().map(|x| x * x).sum::<f64>();
            }
            acc
        });
        // For HF references REST's ctrl sets dfa_hybrid_scf = 0 while the
        // HF eigenvalues already contain the exact exchange, so the static
        // term must use the true hybrid fraction (cf. dft/response.rs).
        self.e[n] + exchange * (1.0 - self.fock_hyb) - self.vxc_nn[n]
    }

    /// Bracket `(lo, hi, t)` of the real-axis grid around `z`, mirroring the
    /// energy path's `ri_gw::interpolate_vchiv_bracket`: the interpolated
    /// operator is `(1-t) M(grid[lo]) + t M(grid[hi])`, and `t = 0` outside the
    /// grid (`lo == hi`).
    fn lr_bracket(&self, z: f64) -> (usize, usize, f64) {
        let zs = &self.z_grid;
        let n = zs.len();
        if n <= 1 || z <= zs[0] {
            return (0, 0, 0.0);
        }
        if z >= zs[n - 1] {
            return (n - 1, n - 1, 0.0);
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if zs[mid] < z {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = ((z - zs[lo]) / (zs[hi] - zs[lo])).clamp(0.0, 1.0);
        (lo, hi, t)
    }

    fn qp_row_table(&self, n: usize) -> QpRowTable {
        let qn = self.with_q(|q| self.gather_row(q, n));
        let m = gemm_nn(&self.s_metric, self.naux, self.naux, &qn, self.naux, self.nmo);
        // Both low-rank slots must be resident **before** the `lr_imag` borrow is
        // taken: `ensure_lr_static` shares the cold path with `ensure_lr` and can
        // rebuild `lr_imag`, which would otherwise panic with
        // `RefCell already borrowed` while `lr_imag` is held below.
        self.ensure_lr();
        self.ensure_lr_static();
        let w_imag: Vec<Vec<f64>> = {
            let lr_imag = self.lr_imag.borrow();
            (0..self.quad.len())
                .map(|wi| lr_imag[wi].response_value_from_qt(&m, self.nmo))
                .collect()
        };
        let w0 = if !self.config.subtracted || self.config.skip_lr_cache {
            Vec::new()
        } else {
            let guard = self.lr_static.borrow();
            guard
                .as_ref()
                .expect("gw_grad: static low-rank factor missing")
                .response_value_from_qt(&m, self.nmo)
        };
        QpRowTable { m, w_imag, w0 }
    }

    /// All pieces of the self-energy at a fixed `omega` from a precomputed
    /// [`QpRowTable`]; optionally collects the response data needed by the
    /// gradient.  Returns `(contour, imag, y_imag, residues)`.
    ///
    /// The value of the self-energy only needs the table's `w_imag` numbers,
    /// which follow from the *element-wise* product of `m` with the low-rank
    /// correction (`y - y0 = s_metric * corr`), so the `y`/`y0` vectors (2
    /// `[naux,nmo]` buffers and one `[naux,naux]` gemm each per quadrature point)
    /// are only built when `want_y` asks for them.
    #[allow(clippy::type_complexity)]
    /// Imaginary-axis response rows `(u, y, y0)` of one orbital row.  AC/WAC
    /// only need these rows; the contour/residue part is handled by their own
    /// Padé machinery.
    fn imag_response(
        &self,
        want_y: bool,
        table: &QpRowTable,
    ) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        if !want_y {
            return Vec::new();
        }
        let y0 = gemm_nn(&self.s_metric, self.naux, self.naux, &table.m, self.naux, self.nmo);
        self.ensure_lr();
        let lr_imag = self.lr_imag.borrow();
        let mut out = Vec::with_capacity(self.quad.len());
        for (wi, &(u, _wgt)) in self.quad.iter().enumerate() {
            let corr = lr_imag[wi].correction_from_qt(&table.m, self.nmo);
            let sc = gemm_nn(&self.s_metric, self.naux, self.naux, &corr, self.naux, self.nmo);
            let y: Vec<f64> = y0.iter().zip(sc.iter()).map(|(a, b)| a + b).collect();
            out.push((u, y, y0.clone()));
        }
        out
    }

    /// Compact interaction storage of a cache: the single metric intermediate
    /// `m = s_metric * q_n`, or empty when the full `y`/`y0` pair is kept.
    ///
    /// `m` does not depend on the quadrature frequency (`y0 = s_metric * m`,
    /// `y = y0 + s_metric * corr(m, u)`), so one `[naux, nmo]` buffer replaces
    /// the `nquad` identical copies the cache used to hold.
    fn compact_m_imag(&self, m: &[f64], compact: bool) -> Vec<f64> {
        if !compact {
            return Vec::new();
        }
        m.to_vec()
    }

    /// Omega-derivative pieces from already-collected response data:
    /// `w_imag[wi][m]` from the imaginary pass and the residue vectors.
    ///
    /// With the static-subtracted functional (`w0` non-empty) the axis kernel
    /// multiplies `W_m(iu) - W_m(0)`, adding `sum_k w_k K'/pi * W_m(0)`; the
    /// static and residue subtractions themselves are omega-independent.
    fn sigma_d_from_parts(
        &self,
        omega: f64,
        w_imag: &[Vec<f64>],
        w0: &[f64],
        residues: &[Residue],
    ) -> f64 {
        let mut d = 0.0f64;
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let w = &w_imag[wi];
            let mut sum_m = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                sum_m += 2.0 * (u * u - t * t) / ((t * t + u * u) * (t * t + u * u)) * w[m];
            }
            d -= sum_m * wgt / TWO_PI;
            if !w0.is_empty() {
                let mut sum_0 = 0.0;
                for m in 0..self.nmo {
                    let t = omega - self.e[m];
                    sum_0 += 2.0 * (u * u - t * t) / ((t * t + u * u) * (t * t + u * u))
                        * w0[m];
                }
                d += sum_0 * wgt / TWO_PI;
            }
        }
        // Residue contribution to `d Sigma / d omega`.  `weight_slope` is
        // `dW_m/dz` of the interpolation rule (the factor is evaluated at the
        // exact `|e_m - omega|` in the exact protocol, so its frequency follows
        // `omega`; a grid node does not move with `omega`, so the nearest rule
        // contributes nothing here), and `z = |e_m - omega|` adds `-zsign`.
        if self.config.exact_residue_z || self.config.low_rank_interp == 1 {
            for r in residues {
                let zsign = if self.e[r.m] - omega >= 0.0 { 1.0 } else { -1.0 };
                d += (-zsign) * r.weight_slope;
            }
        }
        d
    }

    /// `d^2 Sigma_c / d omega^2` at fixed geometry, from already-collected
    /// response data (`w_imag[wi][m]` and the static row `w0`).
    ///
    /// Only the imaginary-axis integral contributes: the residue weights of the
    /// linear interpolation rule have a **constant** slope on each bracketing
    /// interval (the grid nodes do not move with `omega`), and the
    /// exact-residue protocol is rejected by the omega channel of the pullback.
    /// The static `W(0)` subtraction enters through `w0` exactly as in
    /// [`Self::sigma_d_from_parts`].
    fn sigma_d2_from_parts(&self, omega: f64, w_imag: &[Vec<f64>], w0: &[f64]) -> f64 {
        let mut d2 = 0.0f64;
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let w = &w_imag[wi];
            let mut sum_m = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                let dd = t * t + u * u;
                // d^2/d omega^2 [ 2t/(t^2+u^2) ] with `t = omega - e_m`
                sum_m += -4.0 * t * (3.0 * u * u - t * t) / (dd * dd * dd) * w[m];
            }
            d2 -= sum_m * wgt / TWO_PI;
            if !w0.is_empty() {
                let mut sum_0 = 0.0;
                for m in 0..self.nmo {
                    let t = omega - self.e[m];
                    let dd = t * t + u * u;
                    sum_0 += -4.0 * t * (3.0 * u * u - t * t) / (dd * dd * dd) * w0[m];
                }
                d2 += sum_0 * wgt / TWO_PI;
            }
        }
        d2
    }

    /// Solve the QP equation with Newton's method (analytic derivative,
    /// graphical-solver fallback on the jumps of the quantised residue
    /// protocol) and assemble the gradient cache at the converged root.
    pub fn build_target_cache(&self, n: usize) -> QpCache {
        self.build_target_cache_with(n, !self.config.compact_qp_cache)
    }

    /// [`Self::build_target_cache`] with the **compact** interaction storage
    /// ([`QpCache::m_imag`]) regardless of `config.compact_qp_cache`.
    ///
    /// The BSE gradient uses this for every cache it builds: its caches are its
    /// dominant memory term (`nocc + nvir_act` orbitals) and the pullback
    /// rebuilds `y`/`y0` from `m` anyway, so the full `nquad`-copy
    /// representation is never needed there.
    pub fn build_target_cache_compact(&self, n: usize) -> QpCache {
        self.build_target_cache_with(n, false)
    }

    /// Cache of the **static on-shell** quasiparticle energy
    ///
    /// ```text
    /// E = consts_n + Re Sigma_c(e_n),      consts_n = e_n + (1-a)Sigma_x,nn - v_xc,nn
    /// ```
    ///
    /// i.e. MolGW's `EVSC` / `se%nomega = 0` rule: the self-energy is evaluated
    /// **once at the mean-field energy**, there is no root search and no `Z`
    /// renormalisation.  Selected by [`GwGradConfig::static_fallback_threshold`]
    /// for deep levels, and used whenever a root finder cannot bracket a
    /// crossing (which is what the energy path does in both cases).
    ///
    /// The frequency does **not** follow the QP energy, so the pullback must use
    /// [`QpCache::freq`] and the gradient carries the extra
    /// `(d Sigma_c/d omega) * d e_n/dR` term (`dsigma_domega`); see
    /// [`QpCache::static_fallback`].
    /// One-step Newton ("linearized") quasiparticle energy
    ///
    /// ```text
    /// f(w)   = w - consts_n - Sigma_c(w)
    /// E_lin  = e_n - f(e_n)/f'(e_n) = e_n + Z(e_n) [ consts_n + Sigma_c(e_n) - e_n ]
    /// Z(w)   = 1/(1 - dSigma_c/dw)
    /// ```
    ///
    /// Used whenever the root search does not converge, or when the converged
    /// root carries a spectral weight below `GwGradConfig::z_min`.  The
    /// self-energy frequency is the **mean-field** `e_n`, so the analytic
    /// gradient carries the extra `Z^2 * lin_shift * d_R d_omega Sigma_c` term
    /// (the omega-derivative pullback channel) in addition to the usual
    /// `Z`-scaled one.
    /// Shared body of [`Self::build_target_cache`] and
    /// [`Self::build_target_cache_compact`]; `want_y` keeps the full
    /// `y`/`y0` response of every quadrature point in the cache.
    /// Hand in the quasiparticle energies the energy path already solved
    /// (`scf_data.gwqp.0`).  With [`GwGradConfig::pin_qp_from_checkpoint`] set,
    /// every orbital with an entry here is cached **at** that energy instead of
    /// running a second, independent root search.
    pub fn pin_qp_energies(&mut self, qp: Vec<f64>) {
        self.pinned_qp = qp;
    }

    /// Pinned quasiparticle energy of orbital `n`, if any.
    #[inline]
    fn pinned_omega(&self, n: usize) -> Option<f64> {
        if !self.config.pin_qp_from_checkpoint {
            return None;
        }
        match self.pinned_qp.get(n) {
            Some(&w) if w.is_finite() && w != 0.0 => Some(w),
            _ => None,
        }
    }

    /// Response cache of orbital `n` at a **fixed** frequency: no root search, so
    /// the cache, the BSE diagonal and the printed spectrum all refer to the same
    /// quasiparticle solution.  One self-energy evaluation instead of one per
    /// Newton step.
    fn cache_at_omega(&self, n: usize, omega: f64, want_y: bool) -> QpCache {
        assert!(
            matches!(self.config.gw_variant, GwVariant::Ac | GwVariant::Wac),
            "low-rank CD QP caches have been removed; use GwCdGradEngine for CD"
        );
        if self.config.gw_variant == GwVariant::Ac {
            return self.ac_cache_at_omega(n, omega, want_y);
        }
        if self.config.gw_variant == GwVariant::Wac {
            return self.wac_cache_at_omega(n, omega, want_y);
        }
        unreachable!("AC/WAC dispatch returned above")
    }

    fn ac_frequencies(&self) -> Vec<f64> {
        let npts = self.config.ac_num_samples.max(2);
        let idx = crate::ri_gw::ac::get_ac_idx(self.config.num_freq, npts, 2.0 / 3.0);
        idx.iter()
            .map(|&i| {
                let i0 = i
                    .saturating_sub(1)
                    .min(self.quad.len().saturating_sub(1));
                self.quad.get(i0).map(|&(u, _)| u).unwrap_or(0.0)
            })
            .collect()
    }

    /// Fermi shift of the AC sample kernel, `ef = (e_HOMO + e_LUMO)/2`
    /// (the mean-field reference of G0W0, exactly the `gwqp_g` the energy
    /// path hands to `calculate_sigma_c_imag_freq`).
    fn ac_fermi_level(&self) -> f64 {
        let homo = self.nocc.saturating_sub(1);
        let lumo = self.nocc.min(self.nmo.saturating_sub(1));
        0.5 * (self.e[homo] + self.e[lumo])
    }

    /// Sample values `Sigma_c,nn(i lambda_s)` of one orbital row.
    ///
    /// This is the row form of `ri_gw::calculate_sigma_c_imag_freq`: the
    /// screened row `W_c[n,m](iu_k)` is contracted with the same PySCF-style
    /// kernel on the full imaginary-axis quadrature.
    fn ac_setup(&self, table: &QpRowTable) -> AcSetup {
        let lambdas = self.ac_frequencies();
        let ef = self.ac_fermi_level();
        let pi = std::f64::consts::PI;
        let values: Vec<Complex<f64>> = lambdas
            .iter()
            .map(|&lambda| {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
                    for m in 0..self.nmo {
                        let wc = table.w_imag[wi][m];
                        if wc == 0.0 {
                            continue;
                        }
                        let d = Complex::new(ef - self.e[m], lambda);
                        let den = d * d + Complex::new(u * u, 0.0);
                        if den.norm() == 0.0 {
                            continue;
                        }
                        let g0 = d * (wgt / den);
                        re += (-1.0 / pi) * wc * g0.re;
                        im += (-1.0 / pi) * wc * g0.im;
                    }
                }
                Complex::new(re, im)
            })
            .collect();
        AcSetup { lambdas, values }
    }

    // ------------------------------------------------------------------
    // CD-WAC: contour deformation with an analytic continuation of W
    // ------------------------------------------------------------------

    /// Make the CD-WAC reference screening factors resident (once per
    /// geometry).
    ///
    /// The reference set always contains imaginary-axis points (the same
    /// quadrature the CD imaginary integral uses) and, when
    /// [`GwGradConfig::wac_n_real`] is positive, near-real points
    /// `r_i + i*wac_delta` on a quadratic grid between 0 and
    /// [`GwGradConfig::wac_omega_max`].  The spacing rule is selected by
    /// [`GwGradConfig::wac_grid`]; the default quadratic grid puts most points
    /// at the small frequencies that matter for valence residues while still
    /// covering the deep-core range.
    fn ensure_wac_reference(&self) {
        if self.wac_ref.borrow().is_some() {
            return;
        }
        let nquad = self.quad.len();
        let mut imag_index: Vec<usize> = (0..nquad).collect();
        let want_imag = self.config.wac_n_imag;
        if want_imag > 0 && want_imag < nquad {
            // even thinning of the full quadrature grid
            imag_index = (0..want_imag)
                .map(|i| i * (nquad - 1) / (want_imag - 1).max(1))
                .collect();
        }
        let mut z: Vec<Complex<f64>> = imag_index
            .iter()
            .map(|&i| Complex::new(0.0, self.quad[i].0))
            .collect();
        let delta = self.config.wac_delta;
        let n_real = self.config.wac_n_real;
        let pinned = self.config.wac_omega_max > 0.0;
        // Shape of the real-axis reference grid on `[0, 1]`; the physical node
        // is `omega_max * shape[i] + i*delta`.  All shapes are geometry
        // independent, so a pinned `omega_max` keeps the whole grid pinned.
        let shape: Vec<f64> = (0..n_real)
            .map(|i| {
                let t = if n_real <= 1 {
                    0.0
                } else {
                    i as f64 / (n_real - 1) as f64
                };
                match self.config.wac_grid {
                    // linear
                    1 => t,
                    // geometric (dense near zero: most of the valence pole
                    // weight sits below 1 Ha)
                    2 => (4.0 * t).exp_m1() / 4.0f64.exp_m1(),
                    // Chebyshev-Lobatto: dense at both ends of the window
                    3 => 0.5 * (1.0 - (std::f64::consts::PI * t).cos()),
                    // steeper geometric
                    4 => (8.0 * t).exp_m1() / 8.0f64.exp_m1(),
                    // historical quadratic
                    _ => t * t,
                }
            })
            .collect();
        let mut movers = (0usize, self.nmo.saturating_sub(1));
        if n_real > 0 {
            let (omega_max, span) = if pinned {
                (self.config.wac_omega_max, 0.0)
            } else {
                let mut lo = 0usize;
                let mut hi = 0usize;
                for (i, &x) in self.e.iter().enumerate() {
                    if x < self.e[lo] {
                        lo = i;
                    }
                    if x > self.e[hi] {
                        hi = i;
                    }
                }
                movers = (lo, hi);
                let span = self.e[hi] - self.e[lo];
                (span.max(1.0), if span > 1.0 { 1.0 } else { 0.0 })
            };
            for &s_i in shape.iter() {
                z.push(Complex::new(omega_max * s_i, delta));
            }
        }
        let u: Vec<Complex<f64>> = z.iter().map(|&zz| zz * zz).collect();
        let dz_domega_max: Vec<f64> = (0..z.len())
            .map(|p| {
                if p < imag_index.len() || pinned || n_real == 0 {
                    0.0
                } else {
                    let span = self.e[movers.1] - self.e[movers.0];
                    if span > 1.0 {
                        shape[p - imag_index.len()]
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let du_domega_max: Vec<Complex<f64>> = (0..z.len())
            .map(|p| {
                if p < imag_index.len() || pinned || n_real == 0 {
                    Complex::new(0.0, 0.0)
                } else {
                    let t2 = shape[p - imag_index.len()];
                    // the automatic `omega_max = max(span, 1)` only moves when
                    // the span exceeds 1 Ha
                    let span = self.e[movers.1] - self.e[movers.0];
                    if span > 1.0 {
                        2.0 * z[p] * t2
                    } else {
                        Complex::new(0.0, 0.0)
                    }
                }
            })
            .collect();
        let (naux, nov) = (self.naux, self.de_ia.len());
        let lu: Vec<CdLuFactor> = self.with_qtia(|qtia| {
            z.iter()
                .map(|&zz| {
                    let mut chi = wac_chi0(qtia, naux, nov, &self.de_ia, zz);
                    // A = I - chi0
                    for x in chi.data.iter_mut() {
                        *x = -*x;
                    }
                    for d in 0..naux {
                        chi[[d, d]] += Complex::new(1.0, 0.0);
                    }
                    let re: Vec<f64> = chi.data.iter().map(|c| c.re).collect();
                    let im: Vec<f64> = chi.data.iter().map(|c| c.im).collect();
                    let lu = CdLuFactor::factor(&re, &im, naux);
                    lu
                })
                .collect()
        });
        *self.wac_ref.borrow_mut() = Some(WacReference {
            z,
            u,
            lu,
            n_imag: imag_index.len(),
            imag_index,
            du_domega_max,
            omega_max_movers: movers,
            dz_domega_max,
        });
    }

    /// Reference values `W_c,nm(z_p)` of orbital row `n`, laid out
    /// `[n_ref][nmo]`.
    fn wac_values(
        &self,
        table: &QpRowTable,
        r: &WacReference,
    ) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>) {
        let nmo = self.nmo;
        let naux = self.naux;
        let nref = r.z.len();
        let mut out: Vec<Vec<Complex<f64>>> = Vec::with_capacity(nref);
        let mut dout: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); nmo]; nref];
        for &wi in r.imag_index.iter() {
            out.push(
                table.w_imag[wi]
                    .iter()
                    .map(|&w| Complex::new(w, 0.0))
                    .collect(),
            );
        }
        let qn = &table.m;
        // reference-frequency derivative of the response diagonal,
        // d/dz [-4 de/(de^2 - z^2)] = -8 de z/(de^2 - z^2)^2
        let need_node = r.dz_domega_max.iter().any(|&x| x != 0.0);
        let qtia = if need_node {
            self.with_qtia(|q| q.to_vec())
        } else {
            Vec::new()
        };
        for p in r.n_imag..nref {
            let mut y: Vec<Complex<f64>> = qn.iter().map(|&x| Complex::new(x, 0.0)).collect();
            r.lu[p].solve_in_place(&mut y, nmo);
            let mut w = vec![Complex::new(0.0, 0.0); nmo];
            for m in 0..nmo {
                let mut acc = Complex::new(0.0, 0.0);
                for a in 0..naux {
                    let q = qn[a + m * naux];
                    acc += q * (y[a + m * naux] - Complex::new(q, 0.0));
                }
                w[m] = acc;
            }
            if need_node && r.dz_domega_max[p] != 0.0 {
                // d W / d z = y^T (d chi0/dz) y, then chain dz/domega_max
                let z = r.z[p];
                let wts: Vec<Complex<f64>> = self
                    .de_ia
                    .iter()
                    .map(|&de| {
                        let den = Complex::new(de * de, 0.0) - z * z;
                        Complex::new(-8.0 * de, 0.0) * z / (den * den)
                    })
                    .collect();
                let dchi = wac_chi0_weights(&qtia, naux, self.de_ia.len(), &wts);
                let dz = r.dz_domega_max[p];
                let mut dm = vec![Complex::new(0.0, 0.0); nmo];
                for m in 0..nmo {
                    let mut acc = Complex::new(0.0, 0.0);
                    for a in 0..naux {
                        let ya = y[a + m * naux];
                        for b in 0..naux {
                            acc += ya * dchi[[a, b]] * y[b + m * naux];
                        }
                    }
                    dm[m] = acc * dz;
                }
                dout[p] = dm;
            }
            out.push(w);
        }
        (out, dout)
    }

    /// Imaginary-axis contribution of the CD self-energy and its first two
    /// `omega` derivatives (identical to the CD engine's `imag` part).
    fn wac_imag_part(&self, omega: f64, table: &QpRowTable) -> (f64, f64, f64) {
        let mut i0 = 0.0f64;
        let mut i1 = 0.0f64;
        let mut i2 = 0.0f64;
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let w = &table.w_imag[wi];
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                let d = t * t + u * u;
                s0 += 2.0 * t / d * w[m];
                s1 += 2.0 * (u * u - t * t) / (d * d) * w[m];
                s2 += -4.0 * t * (3.0 * u * u - t * t) / (d * d * d) * w[m];
            }
            i0 += s0 * wgt / TWO_PI;
            i1 += s1 * wgt / TWO_PI;
            i2 += s2 * wgt / TWO_PI;
        }
        (i0, i1, i2)
    }

    /// Active CD residues at `omega` and their Padé-interpolated `W` values.
    ///
    /// Returns `(sigma_residue, d/d omega, d^2/d omega^2)`.
    ///
    /// The pole selection mirrors `ri_gw::contour_rayon`: for `omega` above
    /// the Fermi level the virtual poles with `omega - e_a > -res_tol` enter
    /// with weight `+pole_factor`, below it the occupied poles with
    /// `e_i - omega > -res_tol` enter with weight `-pole_factor`.
    fn wac_residue(
        &self,
        omega: f64,
        n: usize,
        r: &WacReference,
        vals: &[Vec<Complex<f64>>],
        dvals: &[Vec<Complex<f64>>],
    ) -> (f64, f64, f64, Vec<WacEntry>, f64) {
        let ef = 0.5 * (self.e[self.nocc - 1] + self.e[self.nocc]);
        let sign = if omega > ef { 1.0 } else { -1.0 };
        let eta = self.config.eta;
        let mut r0 = 0.0f64;
        let mut r1 = 0.0f64;
        let mut r2 = 0.0f64;
        let mut entries: Vec<WacEntry> = Vec::new();
        let mut node_coeff = 0.0f64;
        let start = if sign > 0.0 { self.nocc } else { 0 };
        let end = if sign > 0.0 { self.nmo } else { self.nocc };
        for m in start..end {
            let de = if sign > 0.0 {
                omega - self.e[m]
            } else {
                self.e[m] - omega
            };
            if de <= -self.config.res_tol {
                continue;
            }
            let pole_factor = if de.abs() < self.config.res_tol { 0.5 } else { 1.0 };
            let weight = sign * pole_factor;
            let ze = Complex::new(de, eta);
            let ue = ze * ze;
            let column: Vec<Complex<f64>> = vals.iter().map(|row| row[m]).collect();
            let p = match pade_derivatives_greedy(&r.u, &column, ue) {
                Some(p) => p,
                None => {
                    panic!(
                        "gw_grad (WAC): the Padé continuation of W broke down for orbital {n}, \
                         pole m = {m} ({} reference frequencies)",
                        r.u.len()
                    );
                }
            };
            // d(zeta)/d omega = sign, u = (zeta + i eta)^2
            let du = 2.0 * ze * sign;
            let d2u = Complex::new(2.0, 0.0);
            r0 += weight * p.value.re;
            r1 += weight * (p.dvalue_dz * du).re;
            r2 += weight * (p.d2value_dz2 * du * du + p.dvalue_dz * d2u).re;
            // Coefficients of the pullback: `alpha[p] = d Re P / d W(z_p)`
            // and the explicit `d/d e_m` of the evaluation node.
            let nref = vals.len();
            let alpha: Vec<f64> = (0..nref).map(|q| p.dvalue_dparam[q].re).collect();
            // automatic reference grid: d u_q / d omega_max * d P / d u_q
            for q in 0..nref {
                // node position and reference value both move with omega_max
                node_coeff += weight
                    * (p.dvalue_dparam[nref + q] * r.du_domega_max[q]).re;
                node_coeff += weight * (p.dvalue_dparam[q] * dvals[q][m]).re;
            }
            // u = (zeta + i eta)^2, d zeta/d e_m = -sign
            let du_de = -2.0 * ze * sign;
            let beta = weight * (p.dvalue_dz * du_de).re;
            entries.push(WacEntry { m, weight, alpha, beta });
        }
        (r0, r1, r2, entries, node_coeff)
    }

    /// CD-WAC self-energy and its first two `omega` derivatives at fixed
    /// geometry: `Sigma = R_wac - I`.
    fn wac_sigma(
        &self,
        omega: f64,
        n: usize,
        table: &QpRowTable,
        r: &WacReference,
        vals: &[Vec<Complex<f64>>],
        dvals: &[Vec<Complex<f64>>],
    ) -> (f64, f64, f64, Vec<WacEntry>, f64) {
        let (i0, i1, i2) = self.wac_imag_part(omega, table);
        let (r0, r1, r2, entries, node_coeff) = self.wac_residue(omega, n, r, vals, dvals);
        (r0 - i0, r1 - i1, r2 - i2, entries, node_coeff)
    }

    /// Assemble a CD-WAC [`QpCache`].  The residue list is empty (the WAC
    /// residue term is carried by the Padé data) and there is no static
    /// subtraction.
    #[allow(clippy::too_many_arguments)]
    fn wac_finish_cache(
        &self,
        n: usize,
        table: &QpRowTable,
        omega: f64,
        freq: f64,
        z_factor: f64,
        dsigma: f64,
        static_fallback: bool,
        want_y: bool,
        entries: Vec<WacEntry>,
        n_ref: usize,
        node_coeff: f64,
    ) -> QpCache {
        let y_imag = self.imag_response(want_y, table);
        QpCache {
            target: n,
            omega,
            freq,
            z_factor,
            dsigma_domega: dsigma,
            static_fallback,
            linearized: false,
            lin_shift: 0.0,
            w_imag: table.w_imag.clone(),
            y_imag: if want_y { y_imag } else { Vec::new() },
            // the WAC pullback needs `m = S q_n` even for a full cache
            m_imag: table.m.clone(),
            w0: Vec::new(),
            y0_static: Vec::new(),
            residues: Vec::new(),
            ac: AcQpData::default(),
            wac: Some(WacQpData { entries, n_ref, node_coeff }),
        }
    }

    /// CD-WAC cache at a fixed frequency (pinned QP energies).
    fn wac_cache_at_omega(&self, n: usize, omega: f64, want_y: bool) -> QpCache {
        let table = self.qp_row_table(n);
        self.ensure_wac_reference();
        let guard = self.wac_ref.borrow();
        let r = guard.as_ref().unwrap();
        let (vals, dvals) = self.wac_values(&table, r);
        let (s, ds, _, entries, node_coeff) = self.wac_sigma(omega, n, &table, r, &vals, &dvals);
        let _ = s;
        let z = 1.0 / (1.0 - ds);
        self.wac_finish_cache(n, &table, omega, omega, z, ds, false, want_y, entries, r.z.len(), node_coeff)
    }

    /// CD-WAC counterpart of [`Self::build_target_cache`].
    fn build_target_cache_wac(&self, n: usize, want_y: bool) -> QpCache {
        let consts = self.consts(n);
        let side = if n < self.nocc { -1.0 } else { 1.0 };
        let table = self.qp_row_table(n);
        self.ensure_wac_reference();
        let guard = self.wac_ref.borrow();
        let r = guard.as_ref().unwrap();
        let (vals, dvals) = self.wac_values(&table, r);
        let deeply_occupied =
            self.e[n] < 0.0 && -self.e[n] > self.config.static_fallback_threshold;
        if deeply_occupied || self.config.force_static_fallback.contains(&n) {
            let freq = self.e[n];
            let (s, ds, _, entries, node_coeff) = self.wac_sigma(freq, n, &table, r, &vals, &dvals);
            return self.wac_finish_cache(
                n, &table, consts + s, freq, 1.0, ds, true, want_y, entries, r.z.len(), node_coeff,
            );
        }
        let mut x = self.e[n] + side * 0.02;
        let mut converged = false;
        for _ in 0..self.config.qpe_max_iter {
            let (s, ds, _, _, _) = self.wac_sigma(x, n, &table, r, &vals, &dvals);
            let f = consts + s - x;
            let df = ds - 1.0;
            if !df.is_finite() || df.abs() < 1.0e-14 {
                break;
            }
            let step = -f / df;
            x += step;
            if step.abs() < self.config.qpe_tol {
                converged = true;
                break;
            }
        }
        if !converged {
            println!(
                "warning!!! gw_grad (WAC) Newton did not converge for orbital {n}; using the \
                 static on-shell energy"
            );
            let freq = self.e[n];
            let (s, ds, _, entries, node_coeff) = self.wac_sigma(freq, n, &table, r, &vals, &dvals);
            return self.wac_finish_cache(
                n, &table, consts + s, freq, 1.0, ds, true, want_y, entries, r.z.len(), node_coeff,
            );
        }
        let (_, ds, _, entries, node_coeff) = self.wac_sigma(x, n, &table, r, &vals, &dvals);
        let z = 1.0 / (1.0 - ds);
        if self.config.z_min > 0.0 && z < self.config.z_min {
            println!(
                "warning!!! gw_grad (WAC) orbital {n}: converged Z = {z:.4} is below z_min = \
                 {:.3}; using the static on-shell energy",
                self.config.z_min
            );
            let freq = self.e[n];
            let (s, ds2, _, entries2, node_coeff2) = self.wac_sigma(freq, n, &table, r, &vals, &dvals);
            return self.wac_finish_cache(
                n, &table, consts + s, freq, 1.0, ds2, true, want_y, entries2, r.z.len(), node_coeff2,
            );
        }
        self.wac_finish_cache(n, &table, x, x, z, ds, false, want_y, entries, r.z.len(), node_coeff)
    }

    /// All derivative pieces of the AC self-energy at one frequency.
    ///
    /// Besides the value and the `omega` derivatives this builds the
    /// coefficients of the *linear functional* the pullback consumes:
    /// `d Sigma_c / d W_c[n,m](iu_k)` and the explicit energy derivative
    /// `d Sigma_c / d e_m` (kernel plus the `ef(e_HOMO, e_LUMO)` chain).
    /// The corresponding `omega`-differentiated coefficients feed the
    /// linearized branch's second-order channel.
    fn ac_parts(&self, table: &QpRowTable, setup: &AcSetup, omega: f64) -> Option<AcParts> {
        let ef = self.ac_fermi_level();
        let z = Complex::new(omega, self.config.ac_eta);
        let deriv = pade_derivatives(&setup.lambdas, ef, &setup.values, z)?;
        let pi = std::f64::consts::PI;
        let npts = setup.values.len();
        let np = npts + 1;
        let (nquad, nmo) = (self.quad.len(), self.nmo);
        let mut w_coeff = vec![vec![0.0f64; nmo]; nquad];
        let mut w_coeff_omega = vec![vec![0.0f64; nmo]; nquad];
        let mut be = vec![0.0f64; nmo];
        let mut be_omega = vec![0.0f64; nmo];
        // `ef` enters both the node positions and the sample kernel.
        let mut ef_chain = deriv.dvalue_dparam[np - 1].re;
        let mut ef_chain_omega = deriv.d2value_dz_dparam[np - 1].re;
        for wi in 0..nquad {
            let (u, wgt) = self.quad[wi];
            let u2 = Complex::new(u * u, 0.0);
            for m in 0..nmo {
                let wc = table.w_imag[wi][m];
                if wc == 0.0 {
                    continue;
                }
                let de = ef - self.e[m];
                // `y_s = sum_k (-1/pi) W_nm(iu_k) g0`, so the derivative with
                // respect to `W` carries no `W` factor, while the derivative
                // with respect to the energies carries `W` once (`k_e`).
                let k_w = Complex::new(-1.0 / pi, 0.0);
                let k_e = Complex::new(-wc / pi, 0.0);
                let mut cw = 0.0f64;
                let mut cwo = 0.0f64;
                let mut cded = 0.0f64;
                let mut cded_o = 0.0f64;
                for s in 0..npts {
                    let d = Complex::new(de, setup.lambdas[s]);
                    let den = d * d + u2;
                    let inv = Complex::new(1.0, 0.0) / den;
                    let g0 = d * inv * wgt;
                    // d g0 / d ef = wgt (u^2 - d^2)/den^2; d g0 / d e_m = -that
                    let dg = (u2 - d * d) * inv * inv * wgt;
                    cw += (deriv.dvalue_dparam[s] * k_w * g0).re;
                    cwo += (deriv.d2value_dz_dparam[s] * k_w * g0).re;
                    cded -= (deriv.dvalue_dparam[s] * k_e * dg).re;
                    cded_o -= (deriv.d2value_dz_dparam[s] * k_e * dg).re;
                    ef_chain += (deriv.dvalue_dparam[s] * k_e * dg).re;
                    ef_chain_omega += (deriv.d2value_dz_dparam[s] * k_e * dg).re;
                }
                w_coeff[wi][m] = cw;
                w_coeff_omega[wi][m] = cwo;
                be[m] += cded;
                be_omega[m] += cded_o;
            }
        }
        let homo = self.nocc.saturating_sub(1);
        let lumo = self.nocc.min(nmo.saturating_sub(1));
        be[homo] += 0.5 * ef_chain;
        be[lumo] += 0.5 * ef_chain;
        be_omega[homo] += 0.5 * ef_chain_omega;
        be_omega[lumo] += 0.5 * ef_chain_omega;
        Some(AcParts {
            sigma: deriv.value.re,
            dsigma_domega: deriv.dvalue_dz.re,
            d2sigma_domega2: deriv.d2value_dz2.re,
            w_coeff,
            be,
            w_coeff_omega,
            be_omega,
        })
    }

    /// Assemble a [`QpCache`] from AC pieces.  The static-subtraction fields
    /// (`w0`, `y0_static`, `residues`) are empty: the AC functional contains
    /// no `W(0)` subtraction and no real-axis residues.
    #[allow(clippy::too_many_arguments)]
    fn ac_finish_cache(
        &self,
        n: usize,
        table: &QpRowTable,
        parts: &AcParts,
        omega: f64,
        freq: f64,
        z_factor: f64,
        static_fallback: bool,
        linearized: bool,
        lin_shift: f64,
        want_y: bool,
    ) -> QpCache {
        let y_imag = self.imag_response(want_y, table);
        // Validation masks for the two channels of the AC pullback.
        let mut w_coeff = parts.w_coeff.clone();
        let mut w_coeff_omega = parts.w_coeff_omega.clone();
        let mut be = parts.be.clone();
        let mut be_omega = parts.be_omega.clone();
        if std::env::var("REST_AC_NO_WCOEFF").is_ok() {
            for row in w_coeff.iter_mut() { row.iter_mut().for_each(|v| *v = 0.0); }
            for row in w_coeff_omega.iter_mut() { row.iter_mut().for_each(|v| *v = 0.0); }
        }
        if std::env::var("REST_AC_NO_BE").is_ok() {
            be.iter_mut().for_each(|v| *v = 0.0);
            be_omega.iter_mut().for_each(|v| *v = 0.0);
        }
        // The omega-differentiated coefficients are only read by the
        // second-order channel of the **linearized** branch; dropping them
        // keeps the resident size of the production (root/static) caches at
        // one extra `nquad x nmo` table instead of three.
        if !linearized {
            w_coeff_omega.clear();
            be_omega.clear();
        }
        QpCache {
            target: n,
            omega,
            freq,
            z_factor,
            dsigma_domega: parts.dsigma_domega,
            static_fallback,
            linearized,
            lin_shift,
            w_imag: table.w_imag.clone(),
            y_imag: if want_y { y_imag } else { Vec::new() },
            m_imag: self.compact_m_imag(&table.m, !want_y),
            w0: Vec::new(),
            y0_static: Vec::new(),
            residues: Vec::new(),
            ac: AcQpData {
                w_coeff,
                be,
                w_coeff_omega,
                be_omega,
                d2sigma_domega2: parts.d2sigma_domega2,
            },
            wac: None,
        }
    }

    /// AC cache at a fixed frequency (pinned QP energy, or the final assembly
    /// of a root/branch).
    fn ac_cache_at_omega(&self, n: usize, omega: f64, want_y: bool) -> QpCache {
        let table = self.qp_row_table(n);
        let setup = self.ac_setup(&table);
        let parts = self
            .ac_parts(&table, &setup, omega)
            .expect("gw_grad: AC Padé construction broke down (continued fraction)");
        let z = 1.0 / (1.0 - parts.dsigma_domega);
        self.ac_finish_cache(n, &table, &parts, omega, omega, z, false, false, 0.0, want_y)
    }

    /// Static on-shell AC cache, `E = consts + Re Sigma_c(e_n)`, `Z = 1`.
    fn ac_static_cache(&self, n: usize, consts: f64, table: &QpRowTable, setup: &AcSetup, want_y: bool) -> QpCache {
        let freq = self.e[n];
        let parts = self
            .ac_parts(table, setup, freq)
            .expect("gw_grad: AC Padé construction broke down (continued fraction)");
        let omega = consts + parts.sigma;
        self.ac_finish_cache(n, table, &parts, omega, freq, 1.0, true, false, 0.0, want_y)
    }

    /// Linearized (one-step Newton) AC cache at the mean-field frequency.
    fn ac_linearized_cache(
        &self,
        n: usize,
        consts: f64,
        table: &QpRowTable,
        setup: &AcSetup,
        want_y: bool,
        capped: bool,
    ) -> QpCache {
        let freq = self.e[n];
        let parts = self
            .ac_parts(table, setup, freq)
            .expect("gw_grad: AC Padé construction broke down (continued fraction)");
        let mut z = 1.0 / (1.0 - parts.dsigma_domega);
        if !(z > 0.0) {
            println!(
                "warning!!! gw_grad (AC) orbital {n}: linearized Z = {z:.4} is not positive; \
                 using the static on-shell energy"
            );
            return self.ac_static_cache(n, consts, table, setup, want_y);
        }
        let shift = consts + parts.sigma - freq;
        if capped {
            let cap = self.config.linear_step_cap;
            let step = z * shift;
            if cap > 0.0 && step.abs() > cap {
                z = if shift != 0.0 { cap / shift.abs() * z.signum() } else { 0.0 };
            }
        }
        let omega = freq + z * shift;
        self.ac_finish_cache(n, table, &parts, omega, freq, z, false, true, shift, want_y)
    }

    /// AC counterpart of [`Self::build_target_cache`].
    fn build_target_cache_ac(&self, n: usize, want_y: bool) -> QpCache {
        let consts = self.consts(n);
        let side = if n < self.nocc { -1.0 } else { 1.0 };
        let table = self.qp_row_table(n);
        let setup = self.ac_setup(&table);
        let deeply_occupied =
            self.e[n] < 0.0 && -self.e[n] > self.config.static_fallback_threshold;
        if deeply_occupied || self.config.force_static_fallback.contains(&n) {
            return self.ac_static_cache(n, consts, &table, &setup, want_y);
        }
        if n >= self.nocc && self.e[n] > self.config.linear_fallback_threshold {
            return self.ac_linearized_cache(n, consts, &table, &setup, want_y, true);
        }
        if self.config.force_linearized.contains(&n) {
            return self.ac_linearized_cache(n, consts, &table, &setup, want_y, false);
        }
        let value = |omega: f64| -> f64 {
            match self.ac_parts(&table, &setup, omega) {
                Some(p) => consts + p.sigma - omega,
                None => f64::NAN,
            }
        };
        if self.config.qpe_graphical {
            let (have_crossing, xg) =
                crate::ri_gw::linear_interpolation_solver(&value, self.e[n], side, 21, 0.1);
            if have_crossing {
                return self.ac_cache_at_omega(n, xg, want_y);
            }
            println!(
                "warning!!! gw_grad (AC) graphical solver found no crossing for orbital {n}; \
                 using the linearized (one-step Newton) functional"
            );
            return self.ac_linearized_cache(n, consts, &table, &setup, want_y, true);
        }
        // Newton on the analytic derivative of the Padé QP equation.  The
        // approximant is smooth in `omega`, so this converges quadratically
        // wherever the root is not on top of a Padé pole.
        let mut x = self.e[n] + side * 0.02;
        let mut converged = false;
        for _ in 0..self.config.qpe_max_iter {
            let parts = match self.ac_parts(&table, &setup, x) {
                Some(p) => p,
                None => break,
            };
            let f = consts + parts.sigma - x;
            let df = parts.dsigma_domega - 1.0;
            if !df.is_finite() || df.abs() < 1.0e-14 {
                break;
            }
            let step = -f / df;
            x += step;
            if step.abs() < self.config.qpe_tol {
                converged = true;
                break;
            }
        }
        if !converged {
            println!(
                "warning!!! gw_grad (AC) Newton did not converge for orbital {n}; \
                 using the linearized (one-step Newton) functional"
            );
            return self.ac_linearized_cache(n, consts, &table, &setup, want_y, true);
        }
        let parts = self
            .ac_parts(&table, &setup, x)
            .expect("gw_grad: AC Padé construction broke down (continued fraction)");
        let z = 1.0 / (1.0 - parts.dsigma_domega);
        if self.config.z_min > 0.0 && z < self.config.z_min {
            println!(
                "warning!!! gw_grad (AC) orbital {n}: converged Z = {z:.4} is below z_min = \
                 {:.3}; using the linearized (one-step Newton) functional",
                self.config.z_min
            );
            return self.ac_linearized_cache(n, consts, &table, &setup, want_y, true);
        }
        self.ac_finish_cache(n, &table, &parts, x, x, z, false, false, 0.0, want_y)
    }

    fn build_target_cache_with(&self, n: usize, want_y: bool) -> QpCache {
        assert!(
            matches!(self.config.gw_variant, GwVariant::Ac | GwVariant::Wac),
            "low-rank CD QP caches have been removed; use GwCdGradEngine for CD"
        );
        // One quasiparticle solution per orbital, the one the energy path used:
        // take it and evaluate the response there.  This is checked first so the
        // branch cascade below never runs for a pinned orbital.
        if let Some(omega) = self.pinned_omega(n) {
            return self.cache_at_omega(n, omega, want_y);
        }
        if self.config.gw_variant == GwVariant::Ac {
            return self.build_target_cache_ac(n, want_y);
        }
        if self.config.gw_variant == GwVariant::Wac {
            return self.build_target_cache_wac(n, want_y);
        }
        unreachable!("AC/WAC dispatch returned above")
    }

    // ------------------------------------------------------------------
    // VJP: pull weighted QP gradients back onto (q, J, e, B)
    // ------------------------------------------------------------------

    /// Covectors `(bq, bJ, be, bB)` such that for every state k
    /// `d Omega_k/dR = sum bq[k]:dq + sum bJ[k]:dJ + sum be[k][p] eps1[p]
    ///              + sum bB[k][p] b_x[p]`.
    ///

    /// Covectors `(bq, bJ, be, bB)` such that for every state k
    /// `d Omega_k/dR = sum bq[k]:dq + sum bJ[k]:dJ + sum be[k][p] eps1[p]
    ///              + sum bB[k][p] b_x[p]`.
    ///
    /// Streaming form: `pages` yields successive disjoint runs of QP caches
    /// (`(col0, page)`, `col0` = index of the first cache in the page) and `scr`
    /// accumulates the covectors, so a driver that can rebuild a cache on demand
    /// (the BSE gradient) never has to hold all of them at once.  `weights` is
    /// row-major `[nk, nmo]`.
    ///
    /// Call [`Self::qp_pullback_streaming_finish`] once after the last page.
    pub fn qp_pullback_streaming(
        &self,
        weights: &[f64],
        scr: &mut QpPullbackScratch,
        pages: impl Iterator<Item = (usize, Vec<QpCache>)>,
    ) {
        let nk = scr.nk;
        assert_eq!(
            weights.len(),
            nk * self.nmo,
            "qp_pullback_streaming: weight row length must be nk*nmo"
        );
        for (col0, page) in pages {
            if page.is_empty() {
                break;
            }
            let refs: Vec<&QpCache> = page.iter().collect();
            self.qp_pullback_chunk(&refs, weights, col0, scr, false);
        }
    }

    /// Process one page of caches (see [`Self::qp_pullback_streaming`]).
    /// `omega_mode` selects the **omega-derivative** channel: every kernel is
    /// replaced by its `d/d omega` counterpart and the per-row coefficients
    /// become `weights * Z^2 * lin_shift` (non-zero only for `linearized`
    /// caches).  The caller runs the chunk twice for a linearized cache and
    /// combines the two contractions; see the module plan.
    fn qp_pullback_chunk(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
        omega_mode: bool,
    ) {
        let (nk, naux, nmo, nocc, nov) = (scr.nk, scr.naux, scr.nmo, scr.nocc, scr.nov);
        let dbg = |name: &str| std::env::var(name).is_ok();
        let no_be_expl = dbg("REST_GWGRAD_NOBEEXPL");
        let no_exch = dbg("REST_GWGRAD_NOEXCH");
        let no_res = dbg("REST_GWGRAD_NORES");
        let no_imag = dbg("REST_GWGRAD_NOIMAG");
        let no_static = dbg("REST_GWGRAD_NOSTATIC");
        let QpPullbackScratch { bq, bj, be, bb, v_imag, v_static, v_res, z_order, v_wac, .. } = scr;
        // `qia` in matrix form, once per chunk: the residue `dW/dz` projections are
        // then BLAS-2 products (see `residue_dwdz_m`).
        let qia_m = to_mat(&self.qia, naux, self.de_ia.len());

        for (jl, cache) in caches.iter().enumerate() {
            let j = col0 + jl;
            let n = cache.target;
            // `c` of this cache in every state (`0` for orbitals without QP
            // weight, e.g. virtuals above the BSE cutoff) — a cache that is
            // inactive in every state is skipped entirely.
            let cs: Vec<f64> = if omega_mode {
                (0..nk)
                    .map(|k| {
                        if cache.linearized {
                            let z = cache.z_factor;
                            weights[k * nmo + j] * z * z * cache.lin_shift
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                (0..nk).map(|k| weights[k * nmo + j] * cache.z_factor).collect()
            };
            if cs.iter().all(|&c| c == 0.0) {
                continue;
            }
            // Omega-mode `eps1[n]` coefficient of the linearized branch.
            //
            // `dE_lin/dR = Z (C' + d_R Sigma_c) + Z^2 S (d_R d_omega Sigma_c
            //              + Sigma_c'' * d e_n/dR)`,
            // so the omega channel must supply the **total** geometry derivative
            // of `Sigma_c'` at `omega = e_n(R)`: the partial `d_R d_omega Sigma_c`
            // (the omega-differentiated kernels below) plus this explicit chain
            // term, which moves the tangent point `e_n`.  `B` has no
            // omega-dependence, so its `b_x` and exchange channels contribute
            // nothing in omega mode (the standard pass already carries `Z`).
            // AC (Padé) caches replace the CD kernel with the coefficients of
            // the linear functional `Sigma_c = sum_{k,m} c_km W_c[n,m](iu_k)`.
            let ac = cache.ac.is_ac();
            let dsigma2 = if omega_mode && !no_imag {
                if ac {
                    cache.ac.d2sigma_domega2
                } else {
                    self.sigma_d2_from_parts(cache.freq, &cache.w_imag, &cache.w0)
                }
            } else {
                0.0
            };
            for k in 0..nk {
                if cs[k] != 0.0 {
                    if omega_mode {
                        be[k][n] += cs[k] * dsigma2;
                    } else {
                        // d consts/dR = eps1[n] + b_x[n]
                        be[k][n] += cs[k];
                        bb[k][n] += cs[k];
                    }
                    if cache.static_fallback && !omega_mode {
                        // The static on-shell functional `consts_n + Sigma_c(e_n)`
                        // is evaluated at the **mean-field** frequency, which
                        // moves with the perturbation: its gradient adds
                        // `(d Sigma_c/d omega) d e_n/dR` (and carries no `Z`).
                        be[k][n] += weights[k * nmo + j] * cache.dsigma_domega;
                    }
                }
            }
            // packed-pair offsets of this cache's rows: pair `(n, m)` for every
            // `m`, i.e. the base column of each write below
            let cols: Vec<usize> = (0..nmo)
                .map(|m| packed_pair_index(nmo, n, m) * naux)
                .collect();
            let compact = cache.y_imag.is_empty() && !cache.m_imag.is_empty();
            // static-subtracted functional?  (`w0` is filled exactly when the
            // cache was built by a subtracted engine with a non-empty row)
            let sub = !cache.w0.is_empty() && (compact || !cache.y0_static.is_empty());
            // `y0 = s_metric * m` is the same for every quadrature point, so it
            // is rebuilt once per cache instead of once per (state, frequency).
            // The response buffers are `MatrixFull`s because the outer-product
            // accumulation below reads them as gemm operands.
            let mut y_m = MatrixFull::new([naux, nmo], 0.0);
            let mut y0_m = MatrixFull::new([naux, nmo], 0.0);
            if compact {
                let y0_own = gemm_nn(&self.s_metric, naux, naux, &cache.m_imag, naux, nmo);
                y0_m.data.copy_from_slice(&y0_own);
            }
            // ---- subtracted rows (built once per cache) ----
            // `y0s = (J - Q(0))^-1 q_n` is the response row of the static
            // `W(0)` terms, `y0m = J^-1 q_n` their metric partner (the axis
            // and residue terms keep the raw per-term pullback; the whole
            // static-subtraction bookkeeping folds into their coefficient
            // `staticc`).  Compact caches rebuild `y0s` from `m` through the
            // static low-rank factor; full ones read the stored row.
            let mut y0s_m = MatrixFull::new([naux, nmo], 0.0);
            if sub {
                if compact {
                    let y0s = self.lr_static_row_response(&cache.m_imag);
                    y0s_m.data.copy_from_slice(&y0s);
                } else {
                    y0s_m.data.copy_from_slice(&cache.y0_static);
                    y0_m.data.copy_from_slice(&cache.y_imag[0].2);
                }
            }
            // per-m static coefficient of the subtracted functional:
            // `staticc_m = (1/2 - f_m) + A_m - active_m` with
            // `A_m = 1/pi sum_k w_k K(t_m, u_k)` and `active_m` the signed
            // pole factor, so that the pullback of
            // `sum_wi a_wi W_wi + sum_m staticc_m W(0) + sum_res s pf W_res`
            // is exactly the gradient of `Sigma_sub` (module header).  The
            // `no_*` debug masks keep a masked pullback the exact gradient of
            // the masked functional, as in the full-CD `phase_mask`.
            // `staticc_d` is the omega-derivative `d A_m/d omega` of the same
            // coefficient (its `f_term` and `active` parts do not depend on
            // omega); the omega channel reweights `W(0)` by it.
            let (staticc, staticc_d): (Vec<f64>, Vec<f64>) = if sub {
                let mut active = vec![0.0f64; nmo];
                for r in cache.residues.iter() {
                    active[r.m] += r.pole_weight;
                }
                let mut sc = Vec::with_capacity(nmo);
                let mut sc_d = Vec::with_capacity(nmo);
                for m in 0..nmo {
                    let half_minus_f = if m < nocc { -0.5 } else { 0.5 };
                    let t = cache.freq - self.e[m];
                    let mut a_m = 0.0f64;
                    let mut a_d = 0.0f64;
                    for &(u, wgt) in self.quad.iter() {
                        let __d = t * t + u * u;
                        a_m += wgt / TWO_PI * 2.0 * t / __d;
                        // d/d omega [2t/(t^2+u^2)] = 2(u^2-t^2)/(t^2+u^2)^2
                        a_d += wgt / TWO_PI * 2.0 * (u * u - t * t) / (__d * __d);
                    }
                    let f_term = if no_static { 0.0 } else { half_minus_f };
                    let s_a = if no_imag { 0.0 } else { a_m };
                    let act = if no_res { 0.0 } else { active[m] };
                    sc.push(f_term + s_a - act);
                    sc_d.push(if no_imag { 0.0 } else { a_d });
                }
                (sc, sc_d)
            } else {
                (Vec::new(), Vec::new())
            };
            // `ya = diag(b) y`, `y0a = diag(b) y0` of the current frequency with
            // the **unit** weights `b_m` (see the accumulation note below)
            let mut ya = MatrixFull::new([naux, nmo], 0.0);
            let mut y0a = MatrixFull::new([naux, nmo], 0.0);
            let mut g0 = MatrixFull::new([naux, naux], 0.0);
            let mut v_unit = MatrixFull::new([naux, naux], 0.0);
            let mut sc = vec![0.0f64; naux * nmo];
            // the compact path rebuilds each frequency's response from `m`, which
            // reads the (evictable) low-rank factors
            if compact {
                self.ensure_lr();
            }
            // `s_metric` is a flat `[naux, naux]` buffer; converting it once per chunk
            // (rather than once per quadrature point, as the flat-slice `gemm_nn` did)
            // removes one `naux^2` copy per frequency from the response phase.
            let s_metric_m = if compact {
                to_mat(&self.s_metric, naux, naux)
            } else {
                MatrixFull::new([0, 0], 0.0)
            };
            for (wi, &(u, wgt)) in self
                .quad
                .iter()
                .enumerate()
                .take(if no_imag { 0 } else { usize::MAX })
            {
                // `y` and `y0` of this frequency, shared by every state `k`
                let t_resp = Instant::now();
                if compact {
                    // The response is **column separable**: `corr = E diag(v) E^T qt` and
                    // `sc = S corr` both act on each column of the right-hand side
                    // independently, so cutting the `nmo` columns into fixed blocks needs
                    // *no* reduction — every output entry is produced by exactly one block,
                    // with the same accumulation order as the serial product — and the
                    // per-worker scratch is one `naux * RESP_BLK` block (~0.5 MB here).
                    let m_imag = &cache.m_imag;
                    let lr_guard = self.lr_imag.borrow();
                    let factor = &lr_guard[wi];
                    let spans = index_blocks(nmo, RESP_BLK);
                    let parts: Vec<(usize, Vec<f64>)> = spans
                        .par_iter()
                        .map(|&[c0, c1]| {
                            let nc = c1 - c0;
                            let corr =
                                factor.correction_from_qt(&m_imag[c0 * naux..c1 * naux], nc);
                            let cm = to_mat(&corr, naux, nc);
                            let mut out = MatrixFull::new([naux, nc], 0.0);
                            _dgemm_full(&s_metric_m, 'N', &cm, 'N', &mut out, 1.0, 0.0);
                            (c0, out.data)
                        })
                        .collect();
                    for (c0, data) in parts.iter() {
                        let c0 = *c0;
                        let nc = data.len() / naux;
                        for (lc, c) in (c0..c0 + nc).enumerate() {
                            let add = &data[lc * naux..(lc + 1) * naux];
                            sc[c * naux..(c + 1) * naux].copy_from_slice(add);
                            let base = &y0_m.data[c * naux..(c + 1) * naux];
                            let dst = &mut y_m.data[c * naux..(c + 1) * naux];
                            for i in 0..naux {
                                dst[i] = base[i] + add[i];
                            }
                        }
                    }
                } else if !cache.y_imag.is_empty() {
                    let (_u, y, y0) = &cache.y_imag[wi];
                    y_m.data.copy_from_slice(y);
                    y0_m.data.copy_from_slice(y0);
                    for (d, (a, b)) in sc.iter_mut().zip(y.iter().zip(y0.iter())) {
                        *d = a - b;
                    }
                } else {
                    // empty placeholder: no contribution
                    continue;
                }
                if grad_timing() {
                    T_PULL.with(|c| c.borrow_mut()[0] += t_resp.elapsed().as_secs_f64());
                }
                // ---- outer-product accumulation with **unit** weights ----
                // `bj += sum_m a_m (y0_m y0_m^T - y_m y_m^T)` and
                // `V += sum_m a_m y_m y_m^T` (`a_m = c_k b_m`) are linear in the
                // per-state weight `c_k`, so the two Gram matrices are formed
                // once per (cache, frequency) from the `c = 1` coefficients and
                // scaled into every state below.  Writing them as
                // `Y0 diag(b) Y0^T` / `Y diag(b) Y^T` keeps the signed weights
                // inside the second gemm operand, so each needs a single BLAS-3
                // product (the sign-sorted `sqrt(|b|)` form needed four).
                // `V` is exactly the `y y^T` half of `bj`, so `bj` only
                // accumulates the `y0 y0^T` half here and the finish pass
                // subtracts the accumulated `V` matrices.
                let t_gram = Instant::now();
                for m in 0..nmo {
                    let b = if ac {
                        if omega_mode {
                            cache.ac.w_coeff_omega[wi][m]
                        } else {
                            cache.ac.w_coeff[wi][m]
                        }
                    } else {
                        let t = cache.freq - self.e[m];
                        let __d = t * t + u * u;
                        let __k1 = if omega_mode {
                            2.0 * (u * u - t * t) / (__d * __d)
                        } else {
                            2.0 * t / __d
                        };
                        -wgt / TWO_PI * __k1
                    };
                    let base = m * naux;
                    for r in 0..naux {
                        ya.data[base + r] = b * y_m.data[base + r];
                        y0a.data[base + r] = b * y0_m.data[base + r];
                    }
                }
                // Column-separable split of the two Gram matrices.  `g0[:, q]` depends only
                // on column `q` of `y0a` and on the whole of `y0_m`, so cutting the *output*
                // columns needs no reduction and keeps the serial accumulation order over
                // `m` (one BLAS product per block, same `k` range) — the result is
                // bit-for-bit the single-gemm one.  The second operand of a block is
                // gathered into a small contiguous `[nc, nmo]` buffer, so the extra memory
                // is `GRAM_BLK * (nmo + naux)` doubles per worker (~0.7 MB here) and not a
                // second `naux x naux` accumulator per thread.
                let parts: Vec<(usize, Vec<f64>, Vec<f64>)> = index_blocks(naux, GRAM_BLK)
                    .par_iter()
                    .map(|&[q0, q1]| {
                        let nc = q1 - q0;
                        let mut y0b = MatrixFull::new([nc, nmo], 0.0);
                        let mut yb = MatrixFull::new([nc, nmo], 0.0);
                        for (lc, q) in (q0..q1).enumerate() {
                            // `y0b`/`yb` are col-major `[nc, nmo]`, so entry (lc, m)
                            // sits at `lc + m * nc` (NOT `lc * nmo + m`).
                            for m in 0..nmo {
                                y0b.data[lc + m * nc] = y0a.data[q + m * naux];
                                yb.data[lc + m * nc] = ya.data[q + m * naux];
                            }
                        }
                        let mut g = MatrixFull::new([naux, nc], 0.0);
                        _dgemm_full(&y0_m, 'N', &y0b, 'T', &mut g, 1.0, 0.0);
                        let mut v = MatrixFull::new([naux, nc], 0.0);
                        _dgemm_full(&y_m, 'N', &yb, 'T', &mut v, 1.0, 0.0);
                        (q0, g.data, v.data)
                    })
                    .collect();
                for (q0, gd, vd) in parts.iter() {
                    let q0 = *q0;
                    let nc = gd.len() / naux;
                    g0.data[q0 * naux..(q0 + nc) * naux].copy_from_slice(gd);
                    v_unit.data[q0 * naux..(q0 + nc) * naux].copy_from_slice(vd);
                }
                if grad_timing() {
                    T_PULL.with(|c| c.borrow_mut()[1] += t_gram.elapsed().as_secs_f64());
                }
                let t_acc = Instant::now();
                for k in 0..nk {
                    let c = cs[k];
                    if c == 0.0 {
                        continue;
                    }
                    // dq covector of the pairs (n, m) and the explicit eps1 term.
                    // AC caches carry the whole `e_m` derivative (kernel plus the
                    // `ef` chain) in `cache.ac.be`, added once after this loop.
                    if !no_be_expl && !ac {
                        let f = c * wgt / TWO_PI;
                        for m in 0..nmo {
                            let t = cache.freq - self.e[m];
                    let __d = t * t + u * u;
                    let (__k1, __k2) = if omega_mode {
                        (2.0 * (u * u - t * t) / (__d * __d),
                         -4.0 * t * (3.0 * u * u - t * t) / (__d * __d * __d))
                    } else {
                        (2.0 * t / __d, 2.0 * (u * u - t * t) / (__d * __d))
                    };
                            be[k][m] += f * __k2 * cache.w_imag[wi][m];
                        }
                    }
                    // `sc = y - y0` is already the covector's source vector, so
                    // the update reads one stream instead of subtracting two
                    for m in 0..nmo {
                        let a = if ac {
                            c * if omega_mode {
                                cache.ac.w_coeff_omega[wi][m]
                            } else {
                                cache.ac.w_coeff[wi][m]
                            }
                        } else {
                            let t = cache.freq - self.e[m];
                            let __d = t * t + u * u;
                            let __k1 = if omega_mode {
                                2.0 * (u * u - t * t) / (__d * __d)
                            } else {
                                2.0 * t / __d
                            };
                            -c * wgt / TWO_PI * __k1
                        };
                        let base = m * naux;
                        let col_m = cols[m];
                        for p in 0..naux {
                            bq[k][col_m + p] += 2.0 * a * sc[base + p];
                        }
                    }
                    let t_bj = Instant::now();
                    for (d, s) in bj[k].iter_mut().zip(g0.data.iter()) {
                        *d += c * s;
                    }
                    for (d, s) in v_imag[k][wi].iter_mut().zip(v_unit.data.iter()) {
                        *d += c * s;
                    }
                    if grad_timing() {
                        T_PULL.with(|c| c.borrow_mut()[2] += t_bj.elapsed().as_secs_f64());
                        T_PULL.with(|c| c.borrow_mut()[3] += t_acc.elapsed().as_secs_f64());
                    }
                }
            }

            // AC: the explicit mean-field energy channel `d Sigma_c/d e_m`
            // (kernel derivative **plus** the `ef = (e_HOMO + e_LUMO)/2`
            // chain), once per cache rather than once per quadrature point.
            if ac && !no_imag && !no_be_expl {
                for k in 0..nk {
                    let c = cs[k];
                    if c == 0.0 {
                        continue;
                    }
                    let bev = if omega_mode { &cache.ac.be_omega } else { &cache.ac.be };
                    for m in 0..nmo {
                        be[k][m] += c * bev[m];
                    }
                }
            }

            // ---- CD-WAC residue term -------------------------------------
            // Each reference value `W_c,nm(z_p)` contributes through
            // `alpha_{m,p} = d Re P_m / d W_nm(z_p)`; `W_p = q^T(z_p - y0)`
            // with `z_p = S (I - chi0(z_p))^-1 m` and `y0 = S m`.  The
            // imaginary-axis part is the ordinary CD one and was handled above.
            if let Some(wac) = cache.wac.as_ref() {
                if !no_res && !wac.entries.is_empty() {
                    let mmat = &cache.m_imag;
                    let y0 = gemm_nn(&self.s_metric, naux, naux, mmat, naux, nmo);
                    let y0m = to_mat(&y0, naux, nmo);
                    let guard = self.wac_ref.borrow();
                    let wr = guard
                        .as_ref()
                        .expect("gw_grad (WAC): reference screening factors missing in pullback");
                    let nref = wac.n_ref.min(wr.z.len());
                    for p in 0..nref {
                        // z' = A_p^-1 m, then z = S z'
                        let mut zc: Vec<Complex<f64>> =
                            mmat.iter().map(|&x| Complex::new(x, 0.0)).collect();
                        wr.lu[p].solve_in_place(&mut zc, nmo);
                        let zr = gemm_nn(
                            &self.s_metric,
                            naux,
                            naux,
                            &zc.iter().map(|c| c.re).collect::<Vec<_>>(),
                            naux,
                            nmo,
                        );
                        let zi = gemm_nn(
                            &self.s_metric,
                            naux,
                            naux,
                            &zc.iter().map(|c| c.im).collect::<Vec<_>>(),
                            naux,
                            nmo,
                        );
                        let zmat = MatrixFull::from_vec(
                            [naux, nmo],
                            zr.iter()
                                .zip(zi.iter())
                                .map(|(&r, &i)| Complex::new(r, i))
                                .collect(),
                        )
                        .unwrap();
                        for k in 0..nk {
                            let c = cs[k];
                            if c == 0.0 {
                                continue;
                            }
                            let mut am = vec![0.0f64; nmo];
                            for e in wac.entries.iter() {
                                am[e.m] = c * e.weight * e.alpha[p];
                            }
                            // bq: 2 a Re(z_m - y0_m)
                            for e in wac.entries.iter() {
                                let a = am[e.m];
                                if a == 0.0 {
                                    continue;
                                }
                                let col_m = cols[e.m];
                                for q in 0..naux {
                                    let idx = q + e.m * naux;
                                    bq[k][col_m + q] += 2.0 * a * (zr[idx] - y0[idx]);
                                }
                            }
                            // bj += Y0s Y0^T with the per-m coefficients
                            let mut ys = MatrixFull::new([naux, nmo], 0.0);
                            for m in 0..nmo {
                                if am[m] == 0.0 {
                                    continue;
                                }
                                for q in 0..naux {
                                    ys.data[q + m * naux] = am[m] * y0[q + m * naux];
                                }
                            }
                            let mut yy = MatrixFull::new([naux, naux], 0.0);
                            _dgemm_full(&ys, 'N', &y0m, 'T', &mut yy, 1.0, 0.0);
                            for (d, s) in bj[k].iter_mut().zip(yy.data.iter()) {
                                *d += s;
                            }
                            // v_wac += Re(zs z^T)
                            let mut zs = MatrixFull::from_vec(
                                [naux, nmo],
                                (0..naux * nmo)
                                    .map(|idx| {
                                        let m = idx / naux;
                                        if am[m] == 0.0 {
                                            Complex::new(0.0, 0.0)
                                        } else {
                                            zmat.data[idx] * am[m]
                                        }
                                    })
                                    .collect(),
                            )
                            .unwrap();
                            let mut vc = MatrixFull::from_vec(
                                [naux, naux],
                                vec![Complex::new(0.0, 0.0); naux * naux],
                            )
                            .unwrap();
                            vc.lapack_zgemm(
                                &zs,
                                &zmat,
                                'N',
                                'T',
                                Complex::new(1.0, 0.0),
                                Complex::new(0.0, 0.0),
                            );
                            for (d, s) in v_wac[k][p].iter_mut().zip(vc.data.iter()) {
                                *d += s.re;
                            }
                        }
                    }
                }
                // explicit `d/d e_m` of the residue argument
                if !no_res {
                    for k in 0..nk {
                        let c = cs[k];
                        if c == 0.0 {
                            continue;
                        }
                        for e in wac.entries.iter() {
                            be[k][e.m] += c * e.beta;
                        }
                    }
                    // automatic reference grid: `omega_max = e_last - e_first`
                    if wac.node_coeff != 0.0 {
                        let guard = self.wac_ref.borrow();
                        if let Some(wr) = guard.as_ref() {
                            let (lo, hi) = wr.omega_max_movers;
                            for k in 0..nk {
                                let c = cs[k];
                                if c == 0.0 {
                                    continue;
                                }
                                be[k][hi] += c * wac.node_coeff;
                                be[k][lo] -= c * wac.node_coeff;
                            }
                        }
                    }
                }
            }

            let t_post = Instant::now();
            // `dW/dz` of every residue entry, batched (see `residue_dwdz_all`).
            let dwdz_all: Vec<[f64; 2]> = Vec::new();
            for k in 0..nk {
                let c = cs[k];
                if c == 0.0 {
                    continue;
                }
                // ---- static `W(0)` terms of the subtracted functional ----
                // `sum_m c staticc_m W_m(0)` with `W(0) = q^T [(J-Q(0))^-1 -
                // J^-1] q`: the `A(0)^-1` half pulls through the static
                // response row `y0s` (deferred into the `v_static` bucket),
                // the metric half through `y0m = J^-1 q_n`, and the row
                // covector is their difference `2 (y0s - y0m)`.  The kernel
                // sum `A_m(t_m)` inside `staticc` carries the explicit
                // `-A'_m W(0)` eps1 term of the axis kernel.
                // The static `W(0)` bookkeeping carries its own omega-dependence
                // (`d A_m/d omega`), so the omega channel reweights the block by
                // `staticc_d[m]` instead of `staticc[m]` (and differentiates the
                // explicit kernel below).
                let static_w: &[f64] = if omega_mode { &staticc_d } else { &staticc };
                if sub && static_w.iter().any(|&v| v != 0.0) {
                    let mut pre = vec![0.0f64; naux * nmo];
                    let mut prm = vec![0.0f64; naux * nmo];
                    for m in 0..nmo {
                        let csm = c * static_w[m];
                        let base = m * naux;
                        let col_m = cols[m];
                        for p in 0..naux {
                            pre[base + p] = csm * y0s_m.data[base + p];
                            prm[base + p] = csm * y0_m.data[base + p];
                            bq[k][col_m + p] += 2.0 * (pre[base + p] - prm[base + p]);
                        }
                    }
                    let prem = to_mat(&pre, naux, nmo);
                    let prmm = to_mat(&prm, naux, nmo);
                    let ysm = to_mat(&y0s_m.data, naux, nmo);
                    let ymm = to_mat(&y0_m.data, naux, nmo);
                    let mut vr = MatrixFull::new([naux, naux], 0.0);
                    let mut vm = MatrixFull::new([naux, naux], 0.0);
                    // Column-separable split, exactly as in the Gram products: output
                    // column `q` of `vr`/`vm` depends only on **row** `q` of
                    // `ysm`/`ymm`, so the blocks fill disjoint columns and each gathers
                    // its operand rows into a small contiguous `[nc, nmo]` buffer (the
                    // two blocks of a worker share the gather).  The products are the
                    // same BLAS calls with the same `k` range, so the values are the
                    // serial ones up to BLAS's own blocking.
                    let parts: Vec<(usize, Vec<f64>, Vec<f64>)> = index_blocks(naux, GRAM_BLK)
                        .par_iter()
                        .map(|&[q0, q1]| {
                            let nc = q1 - q0;
                            let mut ysb = MatrixFull::new([nc, nmo], 0.0);
                            let mut ymb = MatrixFull::new([nc, nmo], 0.0);
                            for (lc, q) in (q0..q1).enumerate() {
                                for m in 0..nmo {
                                    ysb.data[lc + m * nc] = ysm.data[q + m * naux];
                                    ymb.data[lc + m * nc] = ymm.data[q + m * naux];
                                }
                            }
                            let mut a1 = MatrixFull::new([naux, nc], 0.0);
                            _dgemm_full(&prem, 'N', &ysb, 'T', &mut a1, 1.0, 0.0);
                            let mut a2 = MatrixFull::new([naux, nc], 0.0);
                            _dgemm_full(&prmm, 'N', &ymb, 'T', &mut a2, 1.0, 0.0);
                            (q0, a1.data, a2.data)
                        })
                        .collect();
                    for (q0, a1, a2) in parts.iter() {
                        let q0 = *q0;
                        let nc = a1.len() / naux;
                        vr.data[q0 * naux..(q0 + nc) * naux].copy_from_slice(a1);
                        vm.data[q0 * naux..(q0 + nc) * naux].copy_from_slice(a2);
                    }
                    for (d, (a, b)) in bj[k].iter_mut().zip(vm.data.iter().zip(vr.data.iter()))
                    {
                        *d += a - b;
                    }
                    for (d, s) in v_static[k].iter_mut().zip(vr.data.iter()) {
                        *d += s;
                    }
                    if !no_be_expl {
                        for m in 0..nmo {
                            let t = cache.freq - self.e[m];
                            let mut ap = 0.0f64;
                            for &(u, wgt) in self.quad.iter() {
                                let __d = t * t + u * u;
                                // standard: `d/dt` of the `A_m` kernel
                                // `2t/(t^2+u^2)`; omega: its second derivative
                                let __k2 = if omega_mode {
                                    -4.0 * t * (3.0 * u * u - t * t) / (__d * __d * __d)
                                } else {
                                    2.0 * (u * u - t * t) / (__d * __d)
                                };
                                ap += wgt / TWO_PI * __k2;
                            }
                            be[k][m] -= c * ap * cache.w0[m];
                        }
                    }
                }
                // ---- residues (see `Residue`) ----
                if grad_timing() {
                    static DIAG: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !DIAG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        let nent: usize = cache
                            .residues
                            .iter()
                            .map(|r| r.entries.iter().filter(|e| !e.y.is_empty()).count())
                            .sum();
                        let nz: usize = z_order[k].len();
                        eprintln!(
                            "[gw_grad timing]   pullback diag: residues {}  live entries {}  distinct z {}  nquad {}  naux {} nmo {} nocc {} sub {}",
                            cache.residues.len(),
                            nent,
                            nz,
                            self.quad.len(),
                            naux,
                            nmo,
                            nocc,
                            sub
                        );
                    }
                }
                let t_res0 = Instant::now();
                T_PULL.with(|c| c.borrow_mut()[4] += t_res0.duration_since(t_post).as_secs_f64());
                for (ri, r) in cache
                    .residues
                    .iter()
                    .enumerate()
                    .take(if no_res { 0 } else { usize::MAX })
                {
                    // The factor frequency follows the pole (`z = |e_m - omega|`)
                    // in the exact protocol and the interpolation weight follows
                    // it in the linear one; `weight_slope` is the corresponding
                    // `dW/de_m` divided by `zsign` and is zero for the nearest
                    // (piecewise-constant) rule.
                    let zsign = if self.e[r.m] - cache.freq >= 0.0 { 1.0 } else { -1.0 };
                    if omega_mode {
                        // `d W_m/d omega` of the entry weights of the **linear**
                        // rule: the grid nodes do not move with omega, only the
                        // bracketing weight `t`, so
                        // `d split_lo/d omega = +zsign/(z_hi-z_lo)` and
                        // `d split_hi/d omega = -zsign/(z_hi-z_lo)`, times the
                        // pole weight.  `sum_i dh_i W_i` reproduces
                        // `sigma_d_from_parts`'s `(-zsign) weight_slope`.
                        // The exact protocol would need `d^2 W/dz^2` and is
                        // rejected loudly below; the single-entry rules have
                        // `dh = 0`.
                        assert!(
                            !self.config.exact_residue_z,
                            "gw_grad: the omega-derivative pullback of a linearized \
                             QP cache does not implement the exact-residue protocol \
                             (`exact_residue_z = true`)"
                        );
                    } else if r.weight_slope != 0.0 {
                        be[k][r.m] += c * r.weight_slope * zsign;
                    }
                    // Movement of the real-axis grid nodes with the reference:
                    // `d de_max/dR = 1.05 * sign * (d e_p - d e_m*)/dR`, so the
                    // node-frequency channel lands on the two scan indices.
                    // `dW/dz` is rebuilt here (once per cache and entry) rather
                    // than stored per Newton step.  The exact-residue protocol
                    // never reads the grid.
                    let t_mv = Instant::now();
                    if !self.config.exact_residue_z {
                        if let Some(mv) = self.z_grid_mover.as_ref() {
                            let ent = &r.entries;
                            let grid_slope = if ent.len() == 1 {
                                if omega_mode {
                                    // the clamped/nearest node does not move
                                    // with omega
                                    0.0
                                } else {
                                    r.pole_weight * dwdz_all[ri][0] * ent[0].node_sens
                                }
                            } else {
                                let (elo, ehi) = (&ent[0], &ent[1]);
                                let t = r.interp_t;
                                let (dlo, dhi) = (dwdz_all[ri][0], dwdz_all[ri][1]);
                                if omega_mode {
                                    // `d/dt` of the standard bracket times
                                    // `dt/d omega = -zsign/(z_hi-z_lo)`
                                    let dz = ehi.z - elo.z;
                                    if dz.abs() < 1.0e-30 {
                                        0.0
                                    } else {
                                        r.pole_weight
                                            * (dhi * ehi.node_sens
                                                - dlo * elo.node_sens
                                                - r.secant * (ehi.node_sens - elo.node_sens))
                                            * (-zsign / dz)
                                    }
                                } else {
                                    r.pole_weight
                                        * ((1.0 - t) * dlo * elo.node_sens
                                            + t * dhi * ehi.node_sens
                                            - r.secant
                                                * ((1.0 - t) * elo.node_sens + t * ehi.node_sens))
                                }
                            };
                            if grid_slope != 0.0 {
                                // `de_max` is a **max** over the scan, so on a
                                // symmetry-degenerate tie it is not
                                // differentiable (its central difference is the
                                // average of the tied branches).  The
                                // basis-independent choice is the tie average,
                                // which is exact for the 1- and 2-fold
                                // degeneracies that dominate; a 3-fold tie
                                // leaves a residual of order the shell's
                                // `eps1` spread times the node sensitivity.
                                let g = c * mv.scale * mv.sign * grid_slope;
                                let wp = g / mv.p_set.len() as f64;
                                let wm = g / mv.m_set.len() as f64;
                                for &p in mv.p_set.iter() {
                                    be[k][p] += wp;
                                }
                                for &m in mv.m_set.iter() {
                                    be[k][m] -= wm;
                                }
                            }
                        }
                    }
                    // dq / dJ covectors and the V bucket of pair (n, m); a linear
                    // pole contributes one entry per bracketing grid node.
                    // In omega mode the entry weight is replaced by its
                    // omega-derivative (see above); the single-entry rules are
                    // omega-independent and contribute nothing.
                    T_PULL.with(|c| c.borrow_mut()[7] += t_mv.elapsed().as_secs_f64());
                    let col_m = cols[r.m];
                    let entries_omega: [f64; 2] = if omega_mode && r.entries.len() == 2 {
                        let dz = r.entries[1].z - r.entries[0].z;
                        if dz.abs() < 1.0e-30 {
                            [0.0, 0.0]
                        } else {
                            let inv = r.pole_weight * zsign / dz;
                            [inv, -inv]
                        }
                    } else {
                        [0.0, 0.0]
                    };
                    for (ei, e) in r.entries.iter().enumerate() {
                        if e.y.is_empty() {
                            continue;
                        }
                        let a = if omega_mode {
                            c * if ei < 2 { entries_omega[ei] } else { 0.0 }
                        } else {
                            c * e.weight
                        };
                        if a == 0.0 {
                            continue;
                        }
                        for p in 0..naux {
                            bq[k][col_m + p] += 2.0 * a * (e.y[p] - e.y0[p]);
                        }
                        // The `V` bucket of this pair (one buffer per distinct pole
                        // frequency `z`), created on first use.
                        let idx =
                            match z_order[k].iter().position(|&z| (z - e.z).abs() < 1.0e-14) {
                                Some(i) => i,
                                None => {
                                    z_order[k].push(e.z);
                                    v_res[k].push(vec![0.0f64; naux * naux]);
                                    z_order[k].len() - 1
                                }
                            };
                        // Both rank-1 updates write only their own `q`-th column of the
                        // two `naux x naux` buffers, so the columns split across the pool
                        // with **no reduction and no partial accumulator**: each element
                        // is still produced by exactly the same expression, and the sum
                        // over residues still runs in residue order, so the result is
                        // bit-identical to the serial one.  `q` outer / `p` inner also
                        // makes both inner loops unit-stride.
                        let t_r1 = Instant::now();
                        let v = &mut v_res[k][idx];
                        let y0 = &e.y0;
                        let y = &e.y;
                        bj[k].par_chunks_mut(RES_BLK * naux)
                            .zip(v.par_chunks_mut(RES_BLK * naux))
                            .enumerate()
                            .for_each(|(bi, (cb, cv))| {
                                let q0 = bi * RES_BLK;
                                for (lq, (cb, cv)) in
                                    cb.chunks_mut(naux).zip(cv.chunks_mut(naux)).enumerate()
                                {
                                    let q = q0 + lq;
                                    let c1 = y0[q];
                                    let c2 = y[q];
                                    for p in 0..naux {
                                        cb[p] += a * (y0[p] * c1 - y[p] * c2);
                                        cv[p] += a * y[p] * c2;
                                    }
                                }
                            });
                        T_PULL.with(|c| c.borrow_mut()[8] += t_r1.elapsed().as_secs_f64());
                    }
                }

                // (the static exchange term `(1-alpha) Sigma_x,nn` is accumulated
                //  *after* this loop: it is state-independent apart from the scalar
                //  `c`, so the `nocc` metric solves and the Gram product are shared
                //  by every state — see below)
                T_PULL.with(|c| c.borrow_mut()[5] += t_res0.elapsed().as_secs_f64());
            }

            // ---- static term B_n = (1-alpha) Sigma_x,nn - v_xc,nn ----
            // REST `consts = eps + (1-alpha) Sigma_x,nn - v_xc,nn` with
            // `Sigma_x,nn = -sum_i (n i|n i) = -sum_i q_ni^T J^{-1} q_ni`,
            // so the derivative contributes
            //   -(1-alpha) [2 sum_i g_ni^T dq_ni - sum_i g_ni^T dJ g_ni],
            // g_ni = J^{-1} q_ni.  The v_xc relaxation term is not
            // implemented (references with a nonzero v_xc are rejected
            // in `canonical_response`).
            //
            // The `nocc` rank-1 updates of the Gram half are **one** gemm,
            // `sum_i g_i g_i^T = G G^T` with `G = [g_1 ... g_nocc]`:  the scalar
            // form wrote the whole `naux x naux` buffer once per occupied orbital
            // with stride-`naux` stores (4784 x 8.9 MB of cache-missing traffic at
            // vitamin C/def2-TZVP) and re-solved every `q_ni` once per state.
            let t_ex0 = Instant::now();
            if (1.0 - self.fock_hyb).abs() > 1.0e-14 && !no_exch && !omega_mode {
                let mut gmat = MatrixFull::new([naux, nocc], 0.0);
                for i in 0..nocc {
                    let qni = self.with_q(|q| self.gather_pair(q, n, i));
                    let g_ni = self.metric_solve(&qni, 1);
                    gmat.data[i * naux..(i + 1) * naux].copy_from_slice(&g_ni);
                    let col_i = packed_pair_index(nmo, n, i) * naux;
                    for k in 0..nk {
                        let c = cs[k];
                        if c == 0.0 {
                            continue;
                        }
                        for p in 0..naux {
                            bq[k][col_i + p] -= 2.0 * c * (1.0 - self.fock_hyb) * g_ni[p];
                        }
                    }
                }
                let mut gg = MatrixFull::new([naux, naux], 0.0);
                _dgemm_full(&gmat, 'N', &gmat, 'T', &mut gg, 1.0, 0.0);
                for k in 0..nk {
                    let c = cs[k];
                    if c == 0.0 {
                        continue;
                    }
                    let f = c * (1.0 - self.fock_hyb);
                    for (d, s) in bj[k].iter_mut().zip(gg.data.iter()) {
                        *d += f * s;
                    }
                }
            }
            T_PULL.with(|c| c.borrow_mut()[6] += t_ex0.elapsed().as_secs_f64());
        }
    }

    /// Close a streaming pullback: solve the deferred polarizability matrices,
    /// symmetrise the covectors and hand them back.
    pub fn qp_pullback_streaming_finish(
        &self,
        scr: QpPullbackScratch,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let nk = scr.nk;
        let (naux, nmo) = (scr.naux, scr.nmo);
        let dbg = |name: &str| std::env::var(name).is_ok();
        let no_res = dbg("REST_GWGRAD_NORES");
        let no_imag = dbg("REST_GWGRAD_NOIMAG");
        let QpPullbackScratch {
            mut bq,
            mut bj,
            mut be,
            bb,
            v_imag,
            v_static,
            v_res,
            z_order,
            imag_factors,
            v_wac,
            nrow_pair,
            ..
        } = scr;
        let nrow_pair = nrow_pair.max(1).min(nmo);

        // ---- deferred polarizability pullbacks ----
        let t_polar = Instant::now();
        if grad_timing() {
            let [t_r, t_g, t_b, t_a, t_pre, t_res, t_ex, t_mv, t_r1] = T_PULL.with(|c| *c.borrow());
            eprintln!(
                "[gw_grad timing]   qp_pullback: response {:.2}s  gram {:.2}s  bj/v accum {:.2}s  bq/be+rest {:.2}s  | per-cache: prologue {:.2}s  residues {:.2}s  exchange {:.2}s",
                t_r, t_g, t_b, (t_a - t_b).max(0.0), t_pre, t_res, t_ex
            );
            eprintln!(
                "[gw_grad timing]   qp_pullback residues split: dwdz/mover {:.2}s  rank-1 updates {:.2}s",
                t_mv, t_r1
            );
            T_PULL.with(|c| c.borrow_mut().fill(0.0));
        }
        // The buckets of one state are independent, and within one bucket the
        // contraction is a reduction over the transition index, so the deferred
        // polar pullback runs on the pool as well — without a second scratch
        // (see `polar_pullback_parallel`).
        for k in 0..nk {
            for (wi, _) in self.quad.iter().enumerate().take(if no_imag { 0 } else { usize::MAX }) {
                let (d, dd) = &imag_factors[wi];
                self.polar_pullback_parallel(&mut bq, &mut be, k, &v_imag[k][wi], d, dd, true);
            }
            if !no_res {
                for (zi, &z) in z_order[k].iter().enumerate() {
                    let (d, dd) = real_response_and_deriv(&self.de_ia, z, self.config.eta);
                    self.polar_pullback_parallel(&mut bq, &mut be, k, &v_res[k][zi], &d, &dd, true);
                }
            }
            // CD-WAC references: complex `z_p`, response diagonal
            // `d_ia = -4 de/(de^2 - z^2)` and its `de` derivative; `V` is the
            // real part of the accumulated Gram matrix, so only `Re[d]`/`Re[dd]`
            // enter the real covectors.
            if !v_wac.is_empty() {
                let guard = self.wac_ref.borrow();
                if let Some(wr) = guard.as_ref() {
                    for (p, v) in v_wac[k].iter().enumerate() {
                        if p >= wr.z.len() || v.iter().all(|&x| x == 0.0) {
                            continue;
                        }
                        let z = wr.z[p];
                        let mut d = vec![0.0f64; self.de_ia.len()];
                        let mut dd = vec![0.0f64; self.de_ia.len()];
                        for (idx, &de) in self.de_ia.iter().enumerate() {
                            let den = Complex::new(de * de, 0.0) - z * z;
                            let dval = Complex::new(-4.0 * de, 0.0) / den;
                            let ddval = Complex::new(-4.0, 0.0) / den
                                + Complex::new(8.0 * de * de, 0.0) / (den * den);
                            d[idx] = dval.re;
                            dd[idx] = ddval.re;
                        }
                        self.polar_pullback_parallel(&mut bq, &mut be, k, v, &d, &dd, true);
                    }
                }
            }
            // The static bucket is processed whenever it is nonzero: with the
            // `REST_GWGRAD_NOSTATIC` mask the coefficient keeps the axis and
            // residue `W(0)` compensations (`A_m` and `active_m`), so only the
            // fully masked combination leaves it empty.  `Q(0)` carries the
            // same real-axis Lorentzian diagonal at `z = 0` as the production
            // grid's first node.
            if v_static[k].iter().any(|&v| v != 0.0) {
                let (d, dd) = real_response_and_deriv(&self.de_ia, 0.0, self.config.eta);
                self.polar_pullback_parallel(&mut bq, &mut be, k, &v_static[k], &d, &dd, true);
            }
        }

        // The imaginary-axis outer products accumulated `bj += sum a_m y0 y0^T`
        // only; their `y y^T` half is exactly the dQ matrix, which the
        // frequency buckets `v_imag[k][wi]` already hold, so the subtraction is
        // one pass over the (small) `naux^2` covectors instead of a second Gram
        // product per cache and frequency.
        for k in 0..nk {
            for v in v_imag[k].iter() {
                for (d, s) in bj[k].iter_mut().zip(v.iter()) {
                    *d -= s;
                }
            }
            for v in v_wac[k].iter() {
                for (d, s) in bj[k].iter_mut().zip(v.iter()) {
                    *d -= s;
                }
            }
        }

        // The packed pair layout accumulated **both** directions of every pair
        // into one slot (the pullback visits each row of each cache) while the
        // diagonal carries a single contribution.  The minimum-norm symmetric
        // representative the folded consumers expect stores the *average* of
        // the two directions, so halve the off-diagonal pairs — exactly the
        // values `symmetrise_pairs` used to write into the full buffer.
        for k in 0..nk {
            let mut tp = 0usize;
            for col in 0..nrow_pair {
                for row in col..nmo {
                    if row != col {
                        let s = &mut bq[k][tp * naux..(tp + 1) * naux];
                        for v in s.iter_mut() {
                            *v *= 0.5;
                        }
                    }
                    tp += 1;
                }
            }
            debug_assert_eq!(tp, banded_pair_count(nmo, nrow_pair), "banded pair walk");
            symmetrise_square(&mut bj[k], naux);
        }
        if std::env::var("REST_GWGRAD_MEMDBG").is_ok() {
            println!(
                "[memdbg]   pair-covector band: {} of {} packed pairs stored ({nrow_pair}/{} rows, {:.1} MB)",
                banded_pair_count(nmo, nrow_pair),
                packed_pair_count(nmo),
                nmo,
                (nk * naux * banded_pair_count(nmo, nrow_pair)) as f64 * 8.0 / 1048576.0,
            );
        }
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   qp_pullback: deferred polar pullbacks {:8.2}s",
                t_polar.elapsed().as_secs_f64()
            );
        }
        (bq, bj, be, bb)
    }

    /// Reset an existing streaming scratch to zero so it can accumulate another
    /// page (used by the parallel pullback, where each page owns a scratch).
    pub fn qp_pullback_scratch_clear(&self, scr: &mut QpPullbackScratch) {
        // A placeholder scratch ([`QpPullbackScratch::default_empty`], used as
        // the `mem::replace` destination of a parallel worker) has no shapes
        // yet; materialise them first so the cleared scratch is usable.
        if scr.imag_factors.is_empty() || scr.bq.is_empty() || scr.bq[0].is_empty() {
            let nk = scr.nk;
            *scr = self.qp_pullback_scratch_banded(nk, scr.nrow_pair.max(1).min(self.nmo));
        }
        let nk = scr.nk;
        for k in 0..nk {
            scr.bq[k].iter_mut().for_each(|v| *v = 0.0);
            scr.bj[k].iter_mut().for_each(|v| *v = 0.0);
            scr.be[k].iter_mut().for_each(|v| *v = 0.0);
            scr.bb[k].iter_mut().for_each(|v| *v = 0.0);
            scr.v_res[k].clear();
            scr.z_order[k].clear();
            for v in scr.v_wac[k].iter_mut() {
                v.iter_mut().for_each(|x| *x = 0.0);
            }
        }
    }

    /// Reset the **page-local** accumulators of an existing scratch, keeping the
    /// per-state frequency bucketed ones.
    ///
    /// `v_imag`/`v_static` of a parallel pullback worker accumulate over *every*
    /// page of its batch before the partial is merged (see
    /// [`Self::qp_pullback_scratch_merge`]), so a page reset must leave them
    /// alone; `bq`/`bj`/`be`/`bb` and the residue buckets are per-partial and are
    /// zeroed (and their shapes materialised if the scratch is a placeholder).
    pub fn qp_pullback_scratch_reset_page(&self, scr: &mut QpPullbackScratch) {
        if scr.imag_factors.is_empty() || scr.bq.is_empty() || scr.bq[0].is_empty() {
            let nk = scr.nk;
            let nrow = scr.nrow_pair.max(1).min(self.nmo);
            *scr = self.qp_pullback_scratch_banded(nk, nrow);
            return;
        }
        let nk = scr.nk;
        for k in 0..nk {
            scr.bq[k].iter_mut().for_each(|v| *v = 0.0);
            scr.bj[k].iter_mut().for_each(|v| *v = 0.0);
            scr.be[k].iter_mut().for_each(|v| *v = 0.0);
            scr.bb[k].iter_mut().for_each(|v| *v = 0.0);
            scr.v_res[k].clear();
            scr.z_order[k].clear();
            for v in scr.v_wac[k].iter_mut() {
                v.iter_mut().for_each(|x| *x = 0.0);
            }
        }
    }

    /// Merge the per-page partials of the parallel pullback into one scratch.
    ///
    /// The merge is an **ordered sum over pages** with a fixed page layout, so
    /// the accumulation order is a function of the orbital index alone and the
    /// result is bit-for-bit independent of the thread count.  The residue
    /// buckets are re-keyed into the merged `z_order`; a bucket's position in
    /// that list does not enter any floating-point expression (the finish pass
    /// reads each bucket with its own frequency), so a different grouping
    /// order cannot change the result.
    pub fn qp_pullback_scratch_merge(&self, dst: &mut QpPullbackScratch, mut src: QpPullbackScratch) {
        let nk = dst.nk;
        debug_assert!(
            dst.is_shaped(nk),
            "merge: dst not shaped (nk {nk}, naux {})",
            dst.naux
        );
        // An **empty** source (a page without a linearized row contributes
        // nothing through the omega channel) is legal and must be a no-op: the
        // caller merges one per page so that the accumulation order is the page
        // index alone.
        if !src.is_shaped(nk) {
            return;
        }
        for k in 0..nk {
            for (d, s) in dst.bq[k].iter_mut().zip(src.bq[k].iter()) {
                *d += *s;
            }
            for (d, s) in dst.bj[k].iter_mut().zip(src.bj[k].iter()) {
                *d += *s;
            }
            for (d, s) in dst.be[k].iter_mut().zip(src.be[k].iter()) {
                *d += *s;
            }
            for (d, s) in dst.bb[k].iter_mut().zip(src.bb[k].iter()) {
                *d += *s;
            }
            for (d, s) in dst.v_imag[k].iter_mut().zip(src.v_imag[k].iter()) {
                for (x, y) in d.iter_mut().zip(s.iter()) {
                    *x += *y;
                }
            }
            for (d, s) in dst.v_static[k].iter_mut().zip(src.v_static[k].iter()) {
                *d += *s;
            }
            for (vd, vs) in dst.v_wac[k].iter_mut().zip(src.v_wac[k].iter()) {
                for (d, s) in vd.iter_mut().zip(vs.iter()) {
                    *d += *s;
                }
            }
            for (zi, &z) in src.z_order[k].iter().enumerate() {
                let idx = match dst.z_order[k]
                    .iter()
                    .position(|&w| (w - z).abs() < 1.0e-14)
                {
                    Some(i) => i,
                    None => {
                        dst.z_order[k].push(z);
                        dst.v_res[k].push(vec![0.0f64; self.naux * self.naux]);
                        dst.z_order[k].len() - 1
                    }
                };
                for (d, s) in dst.v_res[k][idx].iter_mut().zip(src.v_res[k][zi].iter()) {
                    *d += *s;
                }
            }
        }
    }

    /// Fresh streaming scratch ([`Self::qp_pullback_streaming`]).
    pub fn qp_pullback_scratch(&self, nk: usize) -> QpPullbackScratch {
        QpPullbackScratch::new(self, nk)
    }

    /// Width of the pair band the pullback of `caches`/`weights` writes.
    ///
    /// `qp_pullback_chunk` writes the pairs `(cache.target, m)` for every `m`, and
    /// `polar_pullback_blocks` the occupied-virtual pairs `(i, a)`; the smaller index of any of
    /// them is at most `cache.target` (or `i < nocc`).  So `1 + max target` over the caches with
    /// a nonzero weight is the exact band that has to be stored — a *weight-row length* is **not**
    /// the right bound: in the analytic-gradient entry point the caches are a compacted subset of
    /// the orbitals, so a row index is not an orbital index.
    pub fn pullback_pair_band(&self, caches: &[&QpCache], weights: &[Vec<f64>]) -> usize {
        let nrow = (0..caches.len())
            .filter(|&i| weights.iter().any(|w| w.get(i).copied().unwrap_or(0.0) != 0.0))
            .map(|i| caches[i].target + 1)
            .max()
            .unwrap_or(self.nmo)
            .max(self.nocc);
        nrow.max(1).min(self.nmo)
    }

    /// As [`Self::qp_pullback_scratch`], storing only the pair band
    /// `min(p, q) < nrow_pair` of the pair covector (see [`banded_pair_count`]).
    ///
    /// Callers that know the largest orbital carrying a QP weight — the GW/BSE gradient does,
    /// from its weight rows — use this to keep `naux * (banded pairs)` doubles instead of
    /// `naux * nmo(nmo+1)/2`.
    pub fn qp_pullback_scratch_banded(&self, nk: usize, nrow_pair: usize) -> QpPullbackScratch {
        QpPullbackScratch::new_banded(self, nk, nrow_pair)
    }

    /// Process one page of caches in an open streaming pullback.
    pub fn qp_pullback_chunk_public(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
    ) {
        self.qp_pullback_chunk(caches, weights, col0, scr, false);
    }

    /// Omega-derivative counterpart of [`Self::qp_pullback_chunk_public`].
    /// Accumulates `d/d omega` of the same covectors, weighted by
    /// `weights * Z^2 * lin_shift` and restricted to `linearized` caches.
    pub fn qp_pullback_chunk_omega_public(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
    ) {
        self.qp_pullback_chunk(caches, weights, col0, scr, true);
    }

    /// Bytes of one fully built [`QpCache`] of this engine.
    ///
    /// With `compact_qp_cache` a cache keeps a single `[naux, nmo]` metric
    /// intermediate instead of the `nquad` pairs of `[naux, nmo]` response
    /// vectors, so the page size of the streaming pullback is estimated from the
    /// compact footprint (plus the `w_imag` table and a residue budget).
    pub fn qp_cache_estimate_bytes(&self) -> usize {
        if self.config.compact_qp_cache {
            let nres = 4.min(self.nocc + self.nmo);
            self.naux * self.nmo * 8
                + self.quad.len() * self.nmo * 8
                + nres * 2 * self.naux * 8
        } else {
            qp_cache_bytes_estimate(self.naux, self.nmo, self.quad.len())
        }
    }

    /// Non-streaming [`Self::qp_pullback_streaming`]: every cache resident.
    ///
    /// `weights[k][j]` is the derivative of `Omega_k` with respect to the QP
    /// energy of cache `j`; the `Z` renormalisation of each cache is applied
    /// here.
    pub fn qp_pullback(
        &self,
        caches: &[&QpCache],
        weights: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let nk = weights.len();
        for w in weights {
            assert_eq!(w.len(), caches.len(), "qp_pullback: weight row length mismatch");
        }
        let flat: Vec<f64> = weights.iter().flat_map(|w| w.iter().copied()).collect();
        let t_cache = Instant::now();
        // The full packed width: this entry point returns the covector to callers outside the
        // crate, which pair it with the full-width fold/unpack helpers.  The *banded* storage is
        // opted into by the GW/BSE gradient driver, which owns its scratch and knows the band
        // (`qp_pullback_scratch_banded`, see `pullback_pair_band`).
        let mut scr = QpPullbackScratch::new(self, nk);
        let owned: Vec<QpCache> = caches
            .iter()
            .map(|c| QpCache {
                target: c.target,
                omega: c.omega,
                freq: c.freq,
                z_factor: c.z_factor,
                dsigma_domega: c.dsigma_domega,
                static_fallback: c.static_fallback,
                linearized: c.linearized,
                lin_shift: c.lin_shift,
                w_imag: c.w_imag.clone(),
                y_imag: c.y_imag.clone(),
                m_imag: c.m_imag.clone(),
                w0: c.w0.clone(),
                y0_static: c.y0_static.clone(),
                residues: c.residues.clone(),
                ac: c.ac.clone(),
                wac: c.wac.clone(),
            })
            .collect();
        let refs: Vec<&QpCache> = owned.iter().collect();
        self.qp_pullback_chunk(&refs, &flat, 0, &mut scr, false);
        // Omega-derivative channel of the linearized caches: run the same
        // scratch a second time (its `cs` carries `weights * Z^2 * lin_shift`)
        // instead of keeping a second covector set resident, so the peak
        // memory of a linearized pullback equals the standard one.
        if refs.iter().any(|c| c.linearized) {
            self.qp_pullback_chunk(&refs, &flat, 0, &mut scr, true);
        }
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   qp_pullback: cache loop {:8.2}s",
                t_cache.elapsed().as_secs_f64()
            );
        }
        self.qp_pullback_streaming_finish(scr)
    }

    /// Add the pullback of `Tr[V dQ]` to `bq` and `be`, with
    /// `Q = sum_ia q_ia d_ia q_ia^T` and `deps_ia = deps_a - deps_i`.
    /// Deferred polarizability pullback.  `packed` selects the pair layout of
    /// `bq`: the LR streaming pullback accumulates the packed lower triangle,
    /// the CD-GW pullback the full pair matrix.
    #[allow(clippy::too_many_arguments)]
    fn polar_pullback(
        &self,
        bq: &mut [Vec<f64>],
        be: &mut [Vec<f64>],
        k: usize,
        v: &[f64],
        d: &[f64],
        dd: &[f64],
        packed: bool,
    ) {
        let naux = self.naux;
        let nov = self.de_ia.len();
        let vm = to_mat(v, naux, naux);
        let qiam = to_mat(&self.qia, naux, nov);
        let mut vq = MatrixFull::new([naux, nov], 0.0);
        _dgemm_full(&vm, 'N', &qiam, 'N', &mut vq, 1.0, 0.0);
        for col in 0..nov {
            let i = col % self.nocc;
            let a = self.nocc + col / self.nocc;
            let col_pair = if packed {
                packed_pair_index(self.nmo, i, a)
            } else {
                self.pair_col(i, a)
            } * naux;
            let mut bg = 0.0f64;
            for p in 0..naux {
                let vqv = vq.data[p + col * naux];
                bq[k][col_pair + p] += 2.0 * d[col] * vqv;
                bg += self.qia[p + col * naux] * vqv * dd[col];
            }
            be[k][i] -= bg;
            be[k][a] += bg;
        }
    }

    /// Parallel [`Self::polar_pullback`] over a fixed partition of the
    /// **transition index** `ia`.
    ///
    /// `polar_pullback` is the accumulated `v qia` contraction of the deferred
    /// polarizability buckets.  It is a reduction over the transition index `ia`
    /// whose outputs land on **distinct destinations per column**:
    ///
    /// * `vq[:, col]` is reduced within one block;
    /// * `bq[k][pair(i,a), :] += 2 d[col] vq[:, col]` — each `col` maps to a
    ///   unique pair `(i, a)`, so a block writes only its own pairs;
    /// * `be[k][i] -= bg`, `be[k][a] += bg` — every row receives at most one
    ///   `-` and one `+` per block, so block-local values merge in block order.
    ///
    /// The block layout comes from the index alone ([`POLAR_BLK`]) and the
    /// partials are merged in block order, so the result is independent of the
    /// thread count — and it needs **no extra scratch**: one `naux x POLAR_BLK`
    /// matrix and one `nmo` vector per block (measured at vitamin C/def2-TZVP:
    /// this is the 73 s of the 398 s 8-core gradient that ran on one core).
    #[allow(clippy::too_many_arguments)]
    fn polar_pullback_parallel(
        &self,
        bq: &mut [Vec<f64>],
        be: &mut [Vec<f64>],
        k: usize,
        v: &[f64],
        d: &[f64],
        dd: &[f64],
        packed: bool,
    ) {
        polar_pullback_blocks(
            self.naux,
            self.nmo,
            self.nocc,
            &self.qia,
            &self.de_ia,
            bq,
            be,
            k,
            v,
            d,
            dd,
            packed,
        )
    }

    /// Contract the covectors with one nuclear perturbation: returns    /// Contract the covectors with one nuclear perturbation: returns
    /// `sum_k [ bq_k : qx + bJ_k : dJ + be_k . eps1 + bB_k . b_x ]`.
    pub fn contract_perturbation(
        &self,
        bq: &[Vec<f64>],
        bj: &[Vec<f64>],
        be: &[Vec<f64>],
        bb: &[Vec<f64>],
        qx: &[f64],
        dj: &[f64],
        eps1: &[f64],
        b_x: &[f64],
    ) -> Vec<f64> {
        (0..bq.len())
            .map(|k| {
                bq[k].iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f64>()
                    + bj[k].iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>()
                    + be[k].iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>()
                    + bb[k].iter().zip(b_x.iter()).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect()
    }

    /// Derivative of the raw three-center MO integrals for one perturbation:
    /// `qx = C^T dI C + U^T q + q U` (pair-symmetrised), with `dI` supplied
    /// as a full [naux, nao*nao] buffer.
    pub fn qx_from_u(&self, di: &[f64], u: &[f64]) -> Vec<f64> {
        let n2 = self.nao * self.nao;
        self.qx_from_u_getter(&|p, out| out.copy_from_slice(&di[p * n2..(p + 1) * n2]), u)
    }

    /// Same contraction with `dI` assembled block-by-block from one atom's
    /// sliced derivative integrals — the full `dI` never coexists with `qx`.
    pub fn qx_from_u_blocks(
        &self,
        blocks: &AtomDerivBlocks,
        comp: usize,
        u: &[f64],
    ) -> Vec<f64> {
        self.qx_from_u_getter(&|p, out| self.raw.d_i_block(blocks, comp, p, out), u)
    }

    fn qx_from_u_getter(&self, di_getter: &dyn Fn(usize, &mut [f64]), u: &[f64]) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let nmo = self.nmo;
        let n2 = nao * nao;
        let npair = nmo * nmo;
        // one borrow of `q` for the whole routine (`with_q` would rebuild it if a previous
        // gradient released it)
        self.ensure_q();
        let q_all = self.q.borrow();
        let mut qx = vec![0.0f64; naux * npair];

        // ---- dI part: qx[P, p + q*nmo] = (C^T dI_p C)[p, q], per P gemms ----
        {
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p in 0..naux {
                di_getter(p, &mut ip.data);
                _dgemm_full(&ip, 'N', &self.c_mo, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..npair {
                    qx[p + col * naux] = sl.data[col];
                }
            }
        }
        // ---- U part: U^T q + q U.  q is pair-symmetric, so
        //      `(qU)[p,q] = (U^T q)[q,p]`; the pair-major [nmo, nmo*naux]
        //      view of q is contracted with U^T in auxiliary blocks (a
        //      full-length t1 buffer would double the qx memory footprint)
        //      and the transposed pair is added before symmetrising; qx
        //      stays in its native [naux, nmo*nmo] col-major layout.  The
        //      aux block p occupies the contiguous column range
        //      [p*nmo, (p+1)*nmo) of the pair-major view. ----
        let um = to_mat(u, nmo, nmo);
        let chunk_aux = (4_194_304 / (nmo * nmo)).clamp(1, naux);
        let resident_pairmajor = !self.q_pairmajor.is_empty();
        let mut p0 = 0;
        while p0 < naux {
            let n_chunk = (naux - p0).min(chunk_aux);
            let nc = n_chunk * nmo;
            // build the [nmo, n_chunk*nmo] pair-major view of the aux block:
            // element (pi, pj_m + p_l*nmo) = q[pair*naux + p0 + p_l]
            let q2c: Vec<f64> = if resident_pairmajor {
                self.q_pairmajor[p0 * npair..(p0 + n_chunk) * npair].to_vec()
            } else {
                let mut v = vec![0.0f64; nmo * nc];
                for pair in 0..npair {
                    let pi = pair % nmo;
                    let pj_m = pair / nmo;
                    let src = pair * naux + p0;
                    for (p_l, &val) in q_all[src..src + n_chunk].iter().enumerate() {
                        v[pi + (pj_m + p_l * nmo) * nmo] = val;
                    }
                }
                v
            };
            let q2cm = to_mat(&q2c, nmo, nc);
            let mut t1v = MatrixFull::new([nmo, nc], 0.0);
            _dgemm_full(&um, 'T', &q2cm, 'N', &mut t1v, 1.0, 0.0);
            for p in p0..p0 + n_chunk {
                let base = (p - p0) * npair;
                for pj in 0..npair {
                    let pj_swap = (pj % nmo) * nmo + pj / nmo;
                    qx[p + pj * naux] += t1v.data[base + pj] + t1v.data[base + pj_swap];
                }
                // symmetrise the pair block
                for pi in 0..nmo {
                    for pj in (pi + 1)..nmo {
                        let a = pi + pj * nmo;
                        let b = pj + pi * nmo;
                        let s = 0.5 * (qx[p + a * naux] + qx[p + b * naux]);
                        qx[p + a * naux] = s;
                        qx[p + b * naux] = s;
                    }
                }
            }
            p0 += n_chunk;
        }
        qx
    }
}

// ---------------------------------------------------------------------------
// Folded ("AO-basis") evaluation of the qx channel
// ---------------------------------------------------------------------------
//
// The gradient couples the perturbation-independent covector `bq` with the
// perturbation-dependent derivative of the raw three-center MO integrals
//
//     sum_{P,p,q} bq[P,p,q] qx[P,p,q],
//     qx = C^T (dI/dR) C + U^T q + q U .
//
// Materialising `qx` costs two MO transformations of `dI` per perturbation,
// `O(naux nao^2 nmo)` + `O(naux nao nmo^2)`, which summed over the `3 natm`
// perturbations is one power of `N` more than necessary: the mixing of the AO
// and MO bases does not depend on the perturbation at all.
//
// Folding the covector once per geometry removes it:
//
//   * dI channel.  With `B[P] = C bq_P C^T` in the AO basis,
//         sum_{P,pq} bq[P,pq] (C^T dI_P C)[pq]
//       = sum_P sum_{mu,nu} B[P,mu,nu] dI[P,mu,nu] ,
//     an `O(nao^2)` dot per auxiliary function, i.e. `O(naux nao^2)` per
//     perturbation, evaluated from the *same* streamed `[nao,nao]` dI slices
//     (`d_i_block`) that the old route fed into the gemms.
//
//   * orbital-rotation channel.  With `q` and `bq` pair-symmetric,
//         sum_{P,pq} bq[P,pq] (U^T q + q U)[P,pq] = 2 Tr[U^T W],
//         W[r,p] = sum_{P,q} q[P,r,q] bq[P,p,q] ,
//     an `O(nmo^2)` dot per perturbation for a once-per-geometry `O(naux nmo^3)`
//     preparation.
//
