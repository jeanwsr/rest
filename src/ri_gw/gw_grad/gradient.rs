// Total: `O(naux nao^3)` once + `O(3 natm * naux nao^2)` instead of
// `O(3 natm * naux nao^3)`.

/// Perturbation-independent folded form of one `bq` covector (one weight row).
///
/// See the section header above for the derivation; both fields are built by
/// [`GwGradEngine::fold_covector`].
pub struct FoldedCovector {
    /// `B[P, mu + nu*nao] = sum_pq C[mu,p] bq[P,p,q] C[nu,q]`, col-major
    /// `[nao, nao]` per auxiliary function `P`.
    pub b_ao: Vec<f64>,
    /// `W[r + p*nmo] = sum_{P,q} q[P,r,q] bq[P,p,q]`, col-major `[nmo, nmo]`.
    pub w_rot: Vec<f64>,
}

impl GwGradEngine<'_> {
    /// Fold one weight row's `bq` covector into `(B, W)` (see
    /// [`FoldedCovector`]).  Cost `O(naux nao^2 nmo + naux nmo^3)`, paid once
    /// per geometry instead of once per perturbation.
    pub fn fold_covector(&self, bq_k: &[f64]) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let npair = nmo * nmo;
        assert_eq!(bq_k.len(), naux * npair, "fold_covector: bq size mismatch");

        self.ensure_q();
        let q_all = self.q.borrow();
        let mut b_ao = vec![0.0f64; naux * n2];
        let mut x = MatrixFull::new([nao, nmo], 0.0);
        let mut bp = MatrixFull::new([nao, nao], 0.0);
        let mut qm = MatrixFull::new([nmo, nmo], 0.0);
        let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        for p in 0..naux {
            // `q` and `bq` are stored pair-slowest / aux-fastest
            // (`[pair*naux + P]`); gather the auxiliary function's
            // `[nmo, nmo]` block into the contiguous gemm buffers.
            for pair in 0..npair {
                let src = pair * naux + p;
                qm.data[pair] = q_all[src];
                bqm.data[pair] = bq_k[src];
            }
            // W += q_P bq_P  (both pair-symmetric, so this is
            // W[r,p] = sum_q q[P,r,q] bq[P,p,q])
            _dgemm_full(&qm, 'N', &bqm, 'N', &mut wm, 1.0, 1.0);
            // B_P = C bq_P C^T
            _dgemm_full(&self.c_mo, 'N', &bqm, 'N', &mut x, 1.0, 0.0);
            _dgemm_full(&x, 'N', &self.c_mo, 'T', &mut bp, 1.0, 0.0);
            b_ao[p * n2..(p + 1) * n2].copy_from_slice(&bp.data);
        }
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// [`Self::fold_covector_packed`] without reading `q`.
    ///
    /// The orbital-rotation covector is `W = sum_P q_P bq_P = C^T X` with `X = sum_P I_P (C bq_P)`,
    /// which only needs the **raw** three-centre integrals (streamed auxiliary-atom block by block)
    /// instead of the transformed `q` (`naux*nmo^2` doubles, 874 MB at naphthalene/def2-TZVP).
    /// The adjoint gradient uses this variant so that `q` can be released before the fold; `B` is
    /// built with exactly the same operations as in [`Self::fold_covector`] (`Z_P = C bq_P`,
    /// `B_P = Z_P C^T`), so `di_cov` is unchanged, while `W` differs only in the order of the
    /// (mathematically identical) summations.
    pub fn fold_covector_packed_streamed(&self, bq_tp: &[f64]) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let nmo_tp = nmo * (nmo + 1) / 2;
        assert_eq!(bq_tp.len(), naux * nmo_tp, "fold_covector_packed_streamed: bq size mismatch");

        let mut b_ao = vec![0.0f64; naux * n2];
        let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
        let mut z = MatrixFull::new([nao, nmo], 0.0);
        let mut bp = MatrixFull::new([nao, nao], 0.0);
        let mut x_acc = MatrixFull::new([nao, nmo], 0.0);
        // one streaming pass over the auxiliary-atom integral blocks serves every `P`
        self.raw.for_each_i3_atom_block(|p0, nb, blk| {
            for pl in 0..nb {
                let p = p0 + pl;
                // rebuild the symmetric `bq` plane from its lower triangle (`bq` was symmetrised)
                let mut tp = 0usize;
                for col in 0..nmo {
                    for row in col..nmo {
                        let val = bq_tp[tp * naux + p];
                        bqm.data[row + col * nmo] = val;
                        bqm.data[col + row * nmo] = val;
                        tp += 1;
                    }
                }
                _dgemm_full(&self.c_mo, 'N', &bqm, 'N', &mut z, 1.0, 0.0); // Z_P = C bq_P
                let ip = to_mat(&blk[pl * n2..(pl + 1) * n2], nao, nao);
                _dgemm_full(&ip, 'N', &z, 'N', &mut x_acc, 1.0, 1.0); // X += I_P Z_P
                _dgemm_full(&z, 'N', &self.c_mo, 'T', &mut bp, 1.0, 0.0); // B_P = Z_P C^T
                b_ao[p * n2..(p + 1) * n2].copy_from_slice(&bp.data);
            }
        });
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        _dgemm_full(&self.c_mo, 'T', &x_acc, 'N', &mut wm, 1.0, 0.0); // W = C^T X
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// Parallel, deterministic [`Self::fold_covector_packed_streamed`].
    ///
    /// The fold is a pure reduction over the auxiliary index: each `P` builds its own
    /// plane of `b_ao` and contributes one term to `X = sum_P I_P Z_P`.  Splitting the
    /// auxiliary **shells** into [`FOLD_BLK`]-sized batches (a layout fixed by the shell
    /// index alone, never by `rayon::current_num_threads()`) and merging the batch-local
    /// `X` accumulators in index order makes the result independent of the thread count —
    /// the same discipline as [`crate::utilities::par_sum`] — while `b_ao` is written
    /// directly, since each batch owns the contiguous slice of `P` it produces.
    ///
    /// `b_ao` is handed to the batches as disjoint `&mut` slices (the batches tile
    /// `0..naux` by whole shells, so the slices are exactly the batches' planes), which
    /// keeps the peak at one `[nao, nmo]` partial per in-flight batch on top of the
    /// `naux*nao^2` output the serial version also holds — no per-batch copy of the
    /// result.
    ///
    /// The extra `[nao, nao]` block the serial stream already keeps is unchanged: each
    /// batch materialises one auxiliary shell's `int3c2e` block at a time.
    pub fn fold_covector_packed_streamed_parallel(&self, bq_tp: &[f64]) -> FoldedCovector {
        self.fold_covector_packed_streamed_parallel_banded(bq_tp, self.nmo)
    }

    /// [`Self::fold_covector_packed_streamed_parallel`] for a **banded** pair covector, i.e. one
    /// that stores only the packed pairs whose smaller index is `< nrow_pair`
    /// (see [`banded_pair_count`]).
    ///
    /// The band is a prefix of the packed layout and the covector is zero outside it (the LR
    /// engine writes it only at pairs reached by a QP-weighted row), so rebuilding the symmetric
    /// `bqm` from the band alone leaves exactly the same zeros elsewhere and the fold result is
    /// unchanged.
    pub fn fold_covector_packed_streamed_parallel_banded(
        &self,
        bq_tp: &[f64],
        nrow_pair: usize,
    ) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let nrow = nrow_pair.max(1).min(nmo);
        let nmo_tp = banded_pair_count(nmo, nrow);
        assert_eq!(bq_tp.len(), naux * nmo_tp, "fold_covector_packed_streamed: bq size mismatch");

        // Batches of *all* auxiliary shells (the fold sums over every centred integral
        // block, not only the perturbed atom's own functions).
        let loc = self.raw.auxmol.ao_loc();
        let nshl = loc.len() - 1;
        let mut batches: Vec<[usize; 2]> = Vec::new();
        {
            let mut a0 = 0usize;
            while a0 < nshl {
                let mut a1 = a0;
                let f0 = loc[a0];
                let mut f1 = f0;
                while a1 < nshl && f1 - f0 < FOLD_BLK {
                    f1 = loc[a1 + 1];
                    a1 += 1;
                }
                if f1 > f0 {
                    batches.push([f0, f1]);
                }
                a0 = a1;
            }
        }
        let mut b_ao = vec![0.0f64; naux * n2];
        // Disjoint `&mut` views of `b_ao`, one per batch: the batches tile `0..naux` in
        // order, so this is a plain `split_at_mut` walk and needs no unsafe code.
        let mut targets: Vec<&mut [f64]> = Vec::with_capacity(batches.len());
        {
            let mut rest: &mut [f64] = &mut b_ao;
            for &[f0, f1] in batches.iter() {
                let (head, tail) = rest.split_at_mut((f1 - f0) * n2);
                targets.push(head);
                rest = tail;
            }
            debug_assert!(rest.is_empty(), "fold batches must tile the auxiliary basis");
        }
        // Each batch returns its partial `X` (`[nao, nmo]`); the merge is an ordered sum,
        // so the rounding order is a function of the batch layout alone.
        let c_mo = &self.c_mo;
        let rmol = &self.raw.mol;
        let raux = &self.raw.auxmol;
        let nbas_mol = rmol.nbas();
        let mut x_acc = vec![0.0f64; nao * nmo];
        let mut targets = targets.into_iter();
        let mut bi = 0usize;
        while bi < batches.len() {
            let hi = (bi + par_cap()).min(batches.len());
            let chunk_targets: Vec<&mut [f64]> = targets.by_ref().take(hi - bi).collect();
            let parts: Vec<Vec<f64>> = batches[bi..hi]
                .par_iter()
                .zip(chunk_targets.into_par_iter())
                .map(|(&[f0, f1], dst)| {
                let nshl_lo = loc.partition_point(|&f| f < f0);
                let nshl_hi = loc.partition_point(|&f| f < f1);
                let mut x_part = MatrixFull::new([nao, nmo], 0.0);
                let mut z = MatrixFull::new([nao, nmo], 0.0);
                let mut bp = MatrixFull::new([nao, nao], 0.0);
                let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
                for shl in nshl_lo..nshl_hi {
                    let (blk, p0, p1) = RawRiTensors::i3_atom_block_pure(
                        rmol,
                        raux,
                        &[0, nbas_mol],
                        nao,
                        nao,
                        &[shl, shl + 1],
                    );
                    if p1 == p0 {
                        continue;
                    }
                    let nb = p1 - p0;
                    for pl in 0..nb {
                        let p = p0 + pl;
                        // rebuild the symmetric `bq` plane from its lower triangle (`bq` was
                        // symmetrised, so the mirrored entries are the same floating-point number)
                        let mut tp = 0usize;
                        for col in 0..nrow {
                            for row in col..nmo {
                                let val = bq_tp[tp * naux + p];
                                bqm.data[row + col * nmo] = val;
                                bqm.data[col + row * nmo] = val;
                                tp += 1;
                            }
                        }
                        _dgemm_full(c_mo, 'N', &bqm, 'N', &mut z, 1.0, 0.0); // Z_P = C bq_P
                        let ip = to_mat(&blk[pl * n2..(pl + 1) * n2], nao, nao);
                        _dgemm_full(&ip, 'N', &z, 'N', &mut x_part, 1.0, 1.0); // X += I_P Z_P
                        _dgemm_full(&z, 'N', c_mo, 'T', &mut bp, 1.0, 0.0); // B_P = Z_P C^T
                        // `dst` is the batch's slice of `b_ao`, so the plane offset is
                        // relative to the batch start (`p0` advances over the batch's
                        // shells, while `pl` restarts at each shell).
                        let off = (p - f0) * n2;
                        dst[off..off + n2].copy_from_slice(&bp.data);
                    }
                }
                    x_part.data
                })
                .collect();
            for xp in parts.iter() {
                for (a, b) in x_acc.iter_mut().zip(xp.iter()) {
                    *a += *b;
                }
            }
            bi = hi;
        }
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        let x = MatrixFull::from_vec([nao, nmo], x_acc).expect("fold: X shape");
        _dgemm_full(&self.c_mo, 'T', &x, 'N', &mut wm, 1.0, 0.0); // W = C^T X
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// [`Self::fold_covector`] for a **pair-packed** `bq` (`[nmo*(nmo+1)/2, naux]`, lower triangle
    /// of the pair-symmetric covector, aux-fastest), as produced by [`pack_pairs_in_place`].
    ///
    /// The symmetric plane is rebuilt from the packed values, so the fold is exact and identical to
    /// the full-buffer version while half the covector memory is resident.
    pub fn fold_covector_packed(&self, bq_tp: &[f64]) -> FoldedCovector {
        self.fold_covector_packed_banded(bq_tp, self.nmo)
    }

    /// [`Self::fold_covector_packed`] for a **banded** pair covector (see
    /// [`banded_pair_count`]): the pairs outside the band are zero, so leaving their `bqm`
    /// entries at zero rebuilds the identical symmetric plane.
    pub fn fold_covector_packed_banded(&self, bq_tp: &[f64], nrow_pair: usize) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let nrow = nrow_pair.max(1).min(nmo);
        let nmo_tp = banded_pair_count(nmo, nrow);
        assert_eq!(bq_tp.len(), naux * nmo_tp, "fold_covector_packed: bq size mismatch");

        self.ensure_q();
        let q_all = self.q.borrow();
        let mut b_ao = vec![0.0f64; naux * n2];
        let mut x = MatrixFull::new([nao, nmo], 0.0);
        let mut bp = MatrixFull::new([nao, nao], 0.0);
        let mut qm = MatrixFull::new([nmo, nmo], 0.0);
        let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        for p in 0..naux {
            for pair in 0..nmo * nmo {
                qm.data[pair] = q_all[pair * naux + p];
            }
            // rebuild the symmetric `bq` plane from its lower triangle (`bq` has been symmetrised,
            // so the mirrored entries are the same floating-point number)
            let mut tp = 0usize;
            for col in 0..nrow {
                for row in col..nmo {
                    let val = bq_tp[tp * naux + p];
                    bqm.data[row + col * nmo] = val;
                    bqm.data[col + row * nmo] = val;
                    tp += 1;
                }
            }
            // W += q_P bq_P, B_P = C bq_P C^T (same order as `fold_covector`)
            _dgemm_full(&qm, 'N', &bqm, 'N', &mut wm, 1.0, 1.0);
            _dgemm_full(&self.c_mo, 'N', &bqm, 'N', &mut x, 1.0, 0.0);
            _dgemm_full(&x, 'N', &self.c_mo, 'T', &mut bp, 1.0, 0.0);
            b_ao[p * n2..(p + 1) * n2].copy_from_slice(&bp.data);
        }
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// Scalar produced by the `qx` channel of one perturbation for one weight
    /// row, evaluated in the folded form — identical (up to round-off) to
    /// `contract_perturbation(bq_k.., .., &qx_from_u_blocks(blocks, comp, u), ..)`
    /// for that row.
    ///
    /// `dj`, `eps1`, `b_x` and the `bJ`/`be`/`bB` covectors are contracted
    /// exactly as in [`GwGradEngine::contract_perturbation`].
    #[allow(clippy::too_many_arguments)]
    pub fn folded_perturbation_gradient(
        &self,
        fold: &FoldedCovector,
        blocks: &AtomDerivBlocks,
        comp: usize,
        u: &[f64],
        bj_k: &[f64],
        be_k: &[f64],
        bb_k: &[f64],
        dj: &[f64],
        eps1: &[f64],
        b_x: &[f64],
    ) -> f64 {
        let n2 = self.nao * self.nao;
        // dI channel: sum_P <B[P], dI[P]>, streamed slice by slice
        let mut ip = vec![0.0f64; n2];
        let mut g = 0.0f64;
        for p in 0..self.naux {
            self.raw.d_i_block(blocks, comp, p, &mut ip);
            let b = &fold.b_ao[p * n2..(p + 1) * n2];
            g += b.iter().zip(ip.iter()).map(|(x, y)| x * y).sum::<f64>();
        }
        // orbital-rotation channel: sum bq:(U^T q + q U) = 2 Tr[U^T W]
        g += 2.0
            * u.iter()
                .zip(fold.w_rot.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        g += bj_k.iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>();
        g += be_k.iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>();
        g += bb_k.iter().zip(b_x.iter()).map(|(a, b)| a * b).sum::<f64>();
        g
    }

    /// `qx` channel of one perturbation for **several** folded covectors at once.
    ///
    /// Same contraction as the `dI`/`U` part of
    /// [`GwGradEngine::folded_perturbation_gradient`], but the atom's sliced
    /// derivative integrals are summed **directly** against the covectors:
    /// with `dI[mu,nu,P] = -d1[mu-r0,nu,P] (mu on the atom)
    ///                      -d1[nu-r0,mu,P] (nu on the atom)
    ///                      -d2[mu,nu,P-p0]  (P on the atom)`
    /// (see [`RawRiTensors::d_i_block`]) no `[nao,nao]` `dI` plane is ever
    /// materialised, and the `d1`/`d2` blocks are read once for every root
    /// instead of once per root — the integral cost is paid once per
    /// perturbation rather than once per perturbation and root.
    ///
    /// Cost per perturbation: `O(naux*nao*nrow_A)` for the two `d1` terms plus
    /// `O(naux_A*nao^2)` for `d2`, against `O(naux*nao^2)` per root for the
    /// streamed `d_i_block` route.
    pub fn folded_qx_multi(
        &self,
        folds: &[FoldedCovector],
        blocks: &AtomDerivBlocks,
        comp: usize,
        u: &[f64],
    ) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let n2 = nao * nao;
        let nrow = blocks.nrow;
        let row0 = blocks.row0;
        let stride1 = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = n2 * blocks.naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];

        let mut out = vec![0.0f64; folds.len()];
        for (fk, fold) in folds.iter().enumerate() {
            let b_ao = &fold.b_ao;
            let mut g = 0.0f64;
            for p in 0..naux {
                let b = &b_ao[p * n2..(p + 1) * n2];
                // mu on the perturbed atom: -sum_{q,nu} B[row0+q, nu] d1[q,nu,P]
                for nu in 0..nao {
                    let bcol = &b[row0 + nu * nao..row0 + nu * nao + nrow];
                    let off = nu * nrow + p * nrow * nao;
                    let mut acc = 0.0f64;
                    for q in 0..nrow {
                        acc += bcol[q] * d1c[off + q];
                    }
                    g -= acc;
                }
                // nu on the perturbed atom: -sum_{q,mu} B[mu, row0+q] d1[q,mu,P]
                for q in 0..nrow {
                    let brow = &b[(row0 + q) * nao..(row0 + q) * nao + nao];
                    let off = q + p * nrow * nao;
                    let mut acc = 0.0f64;
                    for mu in 0..nao {
                        acc += brow[mu] * d1c[off + mu * nrow];
                    }
                    g -= acc;
                }
                // P an auxiliary function of the perturbed atom
                if p >= blocks.aux0 && p < blocks.aux0 + blocks.naux_a {
                    let d2p = &d2c[(p - blocks.aux0) * n2..(p - blocks.aux0 + 1) * n2];
                    let mut acc = 0.0f64;
                    for k in 0..n2 {
                        acc += b[k] * d2p[k];
                    }
                    g -= acc;
                }
            }
            // orbital-rotation channel: sum bq:(U^T q + q U) = 2 Tr[U^T W]
            g += 2.0
                * u.iter()
                    .zip(fold.w_rot.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            out[fk] = g;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// CPHF response and gradient driver
// ---------------------------------------------------------------------------

/// Frozen-density RI Coulomb derivative `d J[dm] / dR` (PySCF
/// `gw_cd_grad._j_deriv_from`).
fn j_deriv_from(engine: &GwGradEngine, di: &[f64], dj: &[f64], dm: &[f64]) -> Vec<f64> {
    engine.raw.with_i3(|i3| {
        j_deriv_from_impl(engine.nao, engine.naux, i3, di, dj, dm, &|b, n| {
            engine.solve_metric_batch(b, n)
        })
    })
}

fn j_deriv_from_impl(
    nao: usize,
    naux: usize,
    i3: &[f64],
    di: &[f64],
    dj: &[f64],
    dm: &[f64],
    solve: &dyn Fn(&mut [f64], usize),
) -> Vec<f64> {
    let n2 = nao * nao;
    // rho[P] = sum_k I[k, P] dm[k]: contract the natural [n2, naux] views (no `to_mat` copies)
    let mut c = gemm_nt(i3, n2, naux, dm, n2, 1);
    let drho = gemm_nt(di, n2, naux, dm, n2, 1);
    solve(&mut c, 1);
    let djc = gemm_nn(dj, naux, naux, &c, naux, 1);
    let mut dc: Vec<f64> = drho.iter().zip(djc.iter()).map(|(a, b)| a - b).collect();
    solve(&mut dc, 1);
    // out[k] = sum_P dI[k, P] c[P] + sum_P I[k, P] dc[P]
    let c_term = gemm_nn(di, n2, naux, &c, naux, 1);
    let dc_term = gemm_nn(i3, n2, naux, &dc, naux, 1);
    c_term.iter().zip(dc_term.iter()).map(|(a, b)| a + b).collect()
}

/// Frozen-orbital RI exchange derivative (PySCF `gw_cd_grad._k_deriv_from`):
/// with `dm = 2 sum_i C_i C_i^T` the derivative of K is
/// `dK = 2 sum_i [dY_i G_i + (dY_i G_i)^T - Y_i J^{-1} dJ G_i]` with
/// `Y_i[mu,P] = (mu i|P)` and `G_i = J^{-1} Y_i^T`.
///
/// All metric solves run through the pre-computed Cholesky factor with
/// `nao*nocc` right-hand sides at once (PySCF solves per occupied orbital;
/// factoring the metric once per perturbation instead of twice per AO
/// column is the main cost saving).
fn k_deriv_from(engine: &GwGradEngine, di: &[f64], dj: &[f64], cocc: &[f64]) -> Vec<f64> {
    engine.raw.with_i3(|i3| {
        k_deriv_from_impl(
            engine.nao,
            engine.naux,
            engine.nocc,
            i3,
            di,
            dj,
            cocc,
            &|b, n| engine.solve_metric_batch(b, n),
        )
    })
}

fn k_deriv_from_impl(
    nao: usize,
    naux: usize,
    nocc: usize,
    i3: &[f64],
    di: &[f64],
    dj: &[f64],
    cocc: &[f64],
    solve: &dyn Fn(&mut [f64], usize),
) -> Vec<f64> {
    let n2 = nao * nao;
    let ncol = nao * nocc; // metric-solve columns (i, mu) = i*nao + mu

    // Y^T and dY^T of all occupied orbitals side by side:
    //   yt[P, i*nao + mu] = sum_nu I3[mu + nu*nao + P*n2] Cocc[nu, i]
    // per auxiliary centre P: block = I3_p @ Cocc ([nao,nao] @ [nao,nocc])
    let cm = to_mat(cocc, nao, nocc);
    let mut yt = vec![0.0f64; naux * ncol];
    let mut dyt = vec![0.0f64; naux * ncol];
    {
        let mut ip = MatrixFull::new([nao, nao], 0.0);
        let mut block = MatrixFull::new([nao, nocc], 0.0);
        for p in 0..naux {
            ip.data.copy_from_slice(&i3[p * n2..(p + 1) * n2]);
            _dgemm_full(&ip, 'N', &cm, 'N', &mut block, 1.0, 0.0);
            for col in 0..ncol {
                yt[p + col * naux] = block.data[col];
            }
            ip.data.copy_from_slice(&di[p * n2..(p + 1) * n2]);
            _dgemm_full(&ip, 'N', &cm, 'N', &mut block, 1.0, 0.0);
            for col in 0..ncol {
                dyt[p + col * naux] = block.data[col];
            }
        }
    }
    // G = J^{-1} Y^T (in yt) and G2 = J^{-1} dJ G, one multi-column solve each
    let yt0 = yt.clone(); // unsolved Y^T, needed for the Y_i G2_i products
    let mut yt_b = yt;
    // one batched triangular solve over all `nao*nocc` columns (BLAS-3): the
    // per-column route ran `ncol` single-RHS `dtrsv` passes over the same
    // `naux^2` factor, which is memory-bound and dominated this routine
    solve(&mut yt_b, ncol);
    let djm = to_mat(dj, naux, naux);
    let g_pre = to_mat(&yt_b, naux, ncol);
    let mut djg = MatrixFull::new([naux, ncol], 0.0);
    _dgemm_full(&djm, 'N', &g_pre, 'N', &mut djg, 1.0, 0.0);
    let mut djg_flat = djg.data;
    solve(&mut djg_flat, ncol);

    // dK = 2 sum_i [dY_i G_i + (dY_i G_i)^T - Y_i G2_i]; the per-occupied
    // blocks are contiguous column blocks of yt/dyt/yt0/djg
    let mut out = vec![0.0f64; n2];
    let mut dk1 = vec![0.0f64; n2];
    let mut term = vec![0.0f64; n2];
    let mut yb = MatrixFull::new([naux, nao], 0.0);
    let mut dyb = MatrixFull::new([naux, nao], 0.0);
    let mut gb = MatrixFull::new([naux, nao], 0.0);
    let mut g2b = MatrixFull::new([naux, nao], 0.0);
    let mut acc = MatrixFull::new([nao, nao], 0.0);
    for i in 0..nocc {
        let off = i * nao * naux;
        dyb.data.copy_from_slice(&dyt[off..off + naux * nao]);
        yb.data.copy_from_slice(&yt0[off..off + naux * nao]);
        gb.data.copy_from_slice(&yt_b[off..off + naux * nao]);
        g2b.data.copy_from_slice(&djg_flat[off..off + naux * nao]);
        // dY_i G_i: dY_i = dyb^T ([nao, naux] = transpose of the [naux, nao] block)
        _dgemm_full(&dyb, 'T', &gb, 'N', &mut acc, 1.0, 0.0);
        dk1.copy_from_slice(&acc.data);
        _dgemm_full(&yb, 'T', &g2b, 'N', &mut acc, 1.0, 0.0);
        term.copy_from_slice(&acc.data);
        for row in 0..nao {
            for col in 0..nao {
                let k = row + col * nao;
                let k_t = col + row * nao;
                out[k] += 2.0 * (dk1[k] + dk1[k_t] - term[k]);
            }
        }
    }
    out
}

impl GwGradEngine<'_> {
    /// Canonical response of every nuclear perturbation `(atm, comp)`,
    /// computed by the `analdrv` CP-HF driver: `RHessSCF` assembles the
    /// dimensionless CP-HF right-hand side and solves `U + resp(U) = rhs`
    /// for all perturbations at once through its block-Krylov solver
    /// (`krylov_block`), with the RI-JK response of `src/ri_jk/hess_r.rs`
    /// built on `rimatr`.
    ///
    /// Returns one `(U [nmo*nmo col-major], eps1 [nmo], b_x [nmo])` per
    /// perturbation, ordered `atm*3 + comp`; `U` and `eps1` follow
    /// `pyscf/gw/gw_cd_grad*.py::_cp_response_atom` and `b_x` is the v_xc
    /// derivative of the static term `B_nn` (zero for an HF reference; see
    /// the module header).
    pub fn canonical_response_batch(
        &self,
    ) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        self.canonical_response_batch_impl()
    }

    fn canonical_response_batch_impl(&self) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        use crate::analdrv::prelude::{
            AnalDrvConfig, HessNucAPI, RHessCoreAPI, RHessElecInteractAPI, RHessHcore,
            RHessOvlp, RHessSCF,
        };
        use crate::analdrv::response::rresp_interface::rscf_resp_interface;
        use crate::ri_jk::hess_r::RHessRIJK;
        use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
        use rstsr::prelude::*;

        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let natm = self.natm;

        // ---- DFT reference pieces of the CP-KS solve and the B_n covector --
        // The static term `B_n = (1-alpha) Sigma_x - v_xc` of the QP equation
        // is differentiated in three channels: the (1-alpha) RI-exchange part
        // through the (bq, bJ) pullback (see `qp_pullback_chunk`), and the
        // v_xc part through the `b_x` covector:
        //   b_x[n] = -[ <n|dv_xc^expl/dR|n> + 2 sum_p U[p,n] v_xc[p,n]
        //               + <n| fxc(D^x) |n> ],
        // the explicit grid derivative (frozen density and MOs), the orbital
        // rotation and the XC-kernel density response of the reference
        // (port of `pyscf/gw/gw_cd_grad.py::_b_nuclear_deriv`).  The same
        // v_xc pieces must also enter the CP-KS right-hand side and the
        // first-order Fock of the Roothaan restoration, so that `U`, `eps1`
        // are the KS response of the actual reference.
        let is_hf = !self.scf.mol.dfadef.is_dfa_scf();
        let vxc_d1_atoms: Option<Vec<Vec<f64>>> = if is_hf {
            None
        } else {
            assert!(
                self.scf.mol.dfadef.omega().is_none(),
                "gw_grad: range-separated hybrid references are not supported"
            );
            let t0 = Instant::now();
            let v = crate::hessian::rks::compute_vxc_h1ao(self.scf);
            if grad_timing() {
                eprintln!(
                    "[gw_grad timing]   cphf: vxc_deriv1 grid sweep {:8.2}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            Some(v)
        };
        let fxc_cache = crate::hessian::rks::prepare_fxc_cache(self.scf);

        // ---- analdrv CP-HF driver with the RI-JK response ----
        let device = crate::utilities::rstsr_util::device();
        let mol = get_cint_mol(&self.scf.mol);
        let aux = get_cint_aux(&self.scf.mol);
        let mo_coeff =
            rt::asarray((&self.c_mo.data, [nao, nmo], &device)).into_contig(ColMajor);
        let mo_occ =
            rt::asarray((&self.scf.occupation[0], [nmo], &device)).into_contig(ColMajor);
        let mo_energy = rt::asarray((&self.e, [nmo], &device)).into_contig(ColMajor);

        let mut ovlp_obj = RHessOvlp::new(&mol, &device);
        let mut hcore_obj = RHessHcore::new(&mol, &device);
        let rimatr = &self
            .scf
            .rimatr
            .as_ref()
            .expect("gw_grad: analdrv CP-HF requires rimatr (use_auxbas + use_ri_symm)")
            .0;
        use crate::utilities::rstsr_util::RestTensorToRstsrViewAPI;
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        let j2c_decomp = crate::ri_jk::get_j2c_decomp(
            &aux,
            &device,
            self.scf.mol.ctrl.j2c_decomp.clone(),
        );
        // the skeleton/response exchange weight follows the reference: the
        // frozen Fock is `h1 + J1 - 0.5*hyb*K1 (+ v_xc^x)` for a hybrid,
        // exactly `J - K` for HF (`fock_hyb` is 1.0 there)
        let mut rijk_obj =
            RHessRIJK::new_with_cderi(&mol, &aux, 1.0, self.fock_hyb, cderi, j2c_decomp);

        let mut config = AnalDrvConfig::default();
        // The adjoint route's accuracy is set by the Z solve, so the Krylov
        // tolerances are part of the gradient configuration (and can be
        // tightened; see `GwGradConfig::cpscf_tol`).
        config.resp.tol = self.config.cpscf_tol;
        config.resp.tol_inflation = self.config.cpscf_tol_inflation;
        // upstream (analdrv refactor): the CP-SCF solve now lives in the shared
        // response object, which `RHessSCF` only borrows.
        let mut resp = rscf_resp_interface(self.scf, &config);
        let nuc_list: Vec<&mut dyn HessNucAPI> = Vec::new();
        let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        // `hess` holds a mutable borrow of `resp`, so it is confined to this
        // scope; the block ends before `resp` is used directly for the solve.
        let (mut rhs_map, t_prep, t_rhs) = {
            let mut hess = RHessSCF::new(
                mo_coeff, mo_occ, mo_energy, &mut ovlp_obj, nuc_list, core_list, el_list,
                &mut resp, &config.nucgrad,
            );
            let t_prep0 = Instant::now();
            hess.make_response_preparation();
            let t_prep = t_prep0.elapsed().as_secs_f64();

            // ---- dimensionless CP-HF right-hand side (analdrv skeleton route)
            //      and one block-Krylov solve over all perturbations ----
            let t_rhs0 = Instant::now();
            let rhs_map = hess.compute_dimless_cpscf_rhs();
            if std::env::var("REST_GWGRAD_RIJK_TIMING").is_ok() {
                for (k, v) in rijk_obj.timing.iter() {
                    eprintln!("[rijk timing] {k:44} {v:8.3}s");
                }
            }
            let t_rhs = t_rhs0.elapsed().as_secs_f64();
            (rhs_map, t_prep, t_rhs)
        };
        // ---- v_xc skeleton of the CP-KS right-hand side ----
        // `compute_dimless_cpscf_rhs` only knows the core + RI-JK skeleton;
        // a DFT reference adds the frozen-density XC derivative, which enters
        // the dimensionless rhs exactly like `f1mo` does:
        // `rhs[vir,occ] -= (C^T v_xc^x C)[vir,occ] / (e_a - e_i + shift)`.
        if let Some(vxc_d1) = vxc_d1_atoms.as_ref() {
            let level_shift = config.resp.level_shift;
            let rhs_tsr_mut = rhs_map.get_mut("rhs").expect("gw_grad: rhs tensor");
            let off = rhs_tsr_mut.offset();
            let rhs_raw = &mut rhs_tsr_mut.raw_mut()[off..];
            for atm in 0..natm {
                for comp in 0..3 {
                    let vxc_d1_ao = vxc_block_ao(&vxc_d1[atm], comp, nao);
                    let t = gemm_nt(&self.c_mo.data, nao, nmo, &vxc_d1_ao, nao, nao);
                    let vxc_d1_mo = gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo);
                    let k = atm * 3 + comp;
                    let base = k * nmo * nocc;
                    for i in 0..nocc {
                        for a in nocc..nmo {
                            let denom = self.e[a] - self.e[i] + level_shift;
                            rhs_raw[base + a + i * nmo] -=
                                vxc_d1_mo[a + i * nmo] / denom;
                        }
                    }
                }
            }
        }
        let t_solve0 = Instant::now();
        let mo1_tsr = resp.solve_dimless_cpscf(rhs_map["rhs"].view()); // [nmo, nocc, 3, natm]
        let t_solve = t_solve0.elapsed().as_secs_f64();
        // the batched Krylov solve was the last user of the CP-KS response object; release the
        // whole-grid AO tables it retained (a pure cache) before the per-perturbation assembly
        if !is_hf {
        }

        // ---- canonical-orbital assembly per perturbation ----
        // The analdrv tensors order perturbations as col-major [.., .., 3, natm],
        // i.e. k_pert = comp + atm*3.  The full-MO skeleton Fock/overlap
        // derivatives needed by the Roothaan restoration (all nmo columns)
        // are not available from the optimized RI-JK hessian object (only
        // `get_deriv1_bra` is implemented), so they come from the raw-RI
        // route here.
        let mo1_data = &mo1_tsr.raw()[mo1_tsr.offset()..];
        let rhs_tsr = &rhs_map["rhs"];
        let n2 = nao * nao;
        // The per-perturbation assembly is embarrassingly parallel: each
        // perturbation owns its `h1`/`s1`/`dI`/`dJ` slabs and its Roothaan
        // restoration, and every shared object is read-only (`SyncGw`).  It is
        // also the dominant cost of the canonical CP-HF route (61 s at
        // naphthalene/def2-TZVP for the skeleton and 116 s for the finish,
        // single-threaded at 1 core), so it runs on the pool.  The `J`/`K`
        // builders inside are rayon-parallel themselves, which nests safely
        // (rayon work-stealing), and they pin the OpenMP/BLAS threads to 1, so a
        // worker's BLAS calls never oversubscribe the pool.
        //
        // `T_TIMER` is thread-local, so each worker reports its own sub-step sums
        // and the caller folds them.
        struct PertOut {
            t_dij: f64,
            t_skel: f64,
            t_fin: f64,
            sub: [f64; 4],
            resp: (Vec<f64>, Vec<f64>, Vec<f64>),
        }
        // The workers below share the engine through `SyncGw`, which is only sound while the
        // lazy caches are **read-only**.  `int3c2e` (`raw.i3`) is the one they reach through
        // `with_i3`, and `ensure_i3` rebuilds it with `borrow_mut` when it is cold: on a cold
        // cache every worker would try to rebuild it at once, which is both a data race on the
        // borrow flag (observed as `RefCell already borrowed`, ammonia/def2-SVP) and `natm*3`
        // redundant full `int3c2e` evaluations.  Build it once here, on this thread, before the
        // pool starts; the workers then only ever take shared borrows.  This is exactly the
        // resident set the serial loop kept (it built `i3` on the first perturbation and never
        // released it), so the peak is unchanged.
        self.raw.ensure_i3();
        let gw = SyncGw(self);
        // bound outside the closure so that it captures plain references instead
        // of the whole engine (`&GwGradEngine` is not `Sync`)
        let scf_ref = &self.scf;
        let c_mo_data = &self.c_mo.data;
        // Bounded fan-out.  Each perturbation materialises its own
        // `int3c2e_ip1`/`ip2` blocks (`nrow_A*nao*naux*3` + `nao^2*naux_A*3` doubles,
        // hundreds of MB on an acene) *and* drives the RI-JK builders, whose per-task
        // buffers are also sized by the number of running workers.  An unbounded
        // fan-out therefore makes the peak of this phase scale with the core count:
        // on the 16-core AMD9654 queue the acene series needed 8 GB (benzene), 31 GB
        // (naphthalene) and 77 GB (anthracene) and tetracene then hit the job's
        // 134 GB memory limit.  Working in fixed chunks of `PAR_CAP` perturbations
        // keeps the extra resident set bounded (the inner RI-JK parallelism still
        // uses the whole pool), and the results are unchanged because every
        // perturbation is independent.
        let mut per_pert: Vec<PertOut> = Vec::with_capacity(natm * 3);
        let mut k0 = 0usize;
        while k0 < natm * 3 {
            let k1 = (k0 + par_cap()).min(natm * 3);
            let part: Vec<PertOut> = {
            use rayon::prelude::*;
            (k0..k1)
                .into_par_iter()
                .map(|k_pert| {
                    let atm = k_pert / 3;
                    let comp = k_pert % 3;
                    T_TIMER.with(|c| c.borrow_mut().fill(0.0));
                    let t0 = Instant::now();
                    let h1r = crate::hessian::rhf::build_hcore_first_deriv(&scf_ref.mol, atm);
                    let s1r = crate::ri_cphf::cphf_solver_pyscf::build_s1ao_deriv(&scf_ref.mol, atm);
                    let mut t_dij = t0.elapsed().as_secs_f64();
                    // shell-sliced derivative integral blocks of this atom, shared
                    // by the three Cartesian components
                    let blocks = gw.d_atom_blocks(atm);
                    let t0 = Instant::now();
                    let di = gw.d_i_from_blocks(&blocks, comp);
                    let dj = gw.d_j_atom(atm, comp);
                    t_dij += t0.elapsed().as_secs_f64();
                    let t0 = Instant::now();
                    // v_xc skeleton of this perturbation in the MO basis (DFT
                    // references): shared by the skeleton Fock and the b_x covector
                    let vxc_d1_mo: Option<Vec<f64>> = vxc_d1_atoms.as_ref().map(|v| {
                        let vao = vxc_block_ao(&v[atm], comp, nao);
                        let t = gemm_nt(c_mo_data, nao, nmo, &vao, nao, nao);
                        gemm_nn(&t, nmo, nao, c_mo_data, nao, nmo)
                    });
                    let (mut f1_skel_mo, s1_mo, jd, kd) =
                        gw.canonical_skeleton_mo(atm, comp, &h1r, &s1r, &di, &dj);
                    if let Some(vd1) = vxc_d1_mo.as_ref() {
                        for (f, v) in f1_skel_mo.iter_mut().zip(vd1.iter()) {
                            *f += v;
                        }
                    }
                    let t_skel = t0.elapsed().as_secs_f64();
                    let mo1_slab = &mo1_data[k_pert * nmo * nocc..(k_pert + 1) * nmo * nocc];
                    let t0 = Instant::now();
                    let resp = gw.canonical_finish(
                        &f1_skel_mo,
                        &s1_mo,
                        mo1_slab,
                        vxc_d1_mo.as_deref(),
                        fxc_cache.as_ref(),
                    );
                    let t_fin = t0.elapsed().as_secs_f64();
                    let sub = T_TIMER.with(|c| *c.borrow());
                    drop((jd, kd, f1_skel_mo, s1_mo));
                    PertOut { t_dij, t_skel, t_fin, sub, resp }
                })
                .collect()
            };
            per_pert.extend(part);
            k0 = k1;
        }
        let mut responses: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::with_capacity(natm * 3);
        let (mut t_dij, mut t_skel, mut t_fin) = (0.0f64, 0.0f64, 0.0f64);
        let mut sub_tot = [0.0f64; 4];
        for p in per_pert {
            t_dij += p.t_dij;
            t_skel += p.t_skel;
            t_fin += p.t_fin;
            for (a, b) in sub_tot.iter_mut().zip(p.sub.iter()) {
                *a += *b;
            }
            responses.push(p.resp);
        }
        T_TIMER.with(|c| *c.borrow_mut() = sub_tot);

        if grad_timing() {
            let [t_jd, t_kd, t_jup, t_kup] = T_TIMER.with(|c| *c.borrow());
            eprintln!(
                "[gw_grad timing]   cphf: prep {:7.2}s  rhs {:7.2}s  solve {:7.2}s | assembly: di/dj+h1/s1 {:7.2}s  skeleton {:7.2}s (j {:.2} k {:.2})  finish {:7.2}s (J {:.2} K {:.2})",
                t_prep, t_rhs, t_solve, t_dij, t_skel, t_jd, t_kd, t_fin, t_jup, t_kup
            );
        }
        let _ = n2;
        responses
    }

    /// Full-MO skeleton Fock and overlap derivatives of one perturbation,
    /// from the raw-RI route: `f1_skel_mo = C^T (dh1/dR + d(J-K skeleton)/dR) C`
    /// and `s1_mo = C^T (dS/dR) C`, both [nmo*nmo] col-major.
    fn canonical_skeleton_mo(
        &self,
        atm: usize,
        comp: usize,
        h1r: &[f64],
        s1r: &[Vec<f64>],
        di: &[f64],
        dj: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let n2 = nao * nao;

        let dm = self.scf.density_matrix[0].data.clone();
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let (jd, kd) = {
            let t0 = Instant::now();
            let jd = j_deriv_from(self, di, dj, &dm);
            timer_add(0, t0);
            let t0 = Instant::now();
            let kd = k_deriv_from(self, di, dj, &cocc);
            timer_add(1, t0);
            (jd, kd)
        };
        // frozen Fock skeleton `h1 + J1 - 0.5*hyb*K1` — the exchange weight
        // follows the reference (`fock_hyb` is 1.0 for HF)
        let vfrozen: Vec<f64> = jd
            .iter()
            .zip(kd.iter())
            .map(|(a, b)| a - 0.5 * self.fock_hyb * b)
            .collect();
        // Fock skeleton in AO, col-major
        let mut f1_ao = vec![0.0f64; n2];
        for k in 0..n2 {
            let i = k % nao;
            let j = k / nao;
            f1_ao[k] = h1r[comp * n2 + i * nao + j] + vfrozen[k];
        }
        let f1_skel_mo = {
            let t = gemm_nt(&self.c_mo.data, nao, nmo, &f1_ao, nao, nao);
            gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
        };
        // overlap derivative in AO, col-major (s1r is row-major)
        let s1_ao: Vec<f64> = (0..n2)
            .map(|k| {
                let i = k % nao;
                let j = k / nao;
                s1r[comp][i * nao + j]
            })
            .collect();
        let s1_mo = {
            let t = gemm_nt(&self.c_mo.data, nao, nmo, &s1_ao, nao, nao);
            gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
        };
        (f1_skel_mo, s1_mo, jd, kd)
    }

    /// Canonical-orbital response `(u, eps1, b_x)` of one perturbation from
    /// the analdrv CP-HF solution: relaxed first-order density, complete
    /// first-order Fock and the Roothaan-style orbital rotations.  With a DFT
    /// reference (`vxc_d1_mo` given) the relaxed Fock carries the XC-kernel
    /// response `fxc(D^x)` and `b_x` collects the v_xc derivative of the
    /// static term `B_n = (1-alpha) Sigma_x - v_xc` (see
    /// `canonical_response_batch_impl`); for HF `b_x` stays zero.
    fn canonical_finish(
        &self,
        f1_skel_mo: &[f64], // skeleton F1 in MO basis [nmo*nmo], col-major
        s1_mo: &[f64],      // overlap derivative in MO basis [nmo*nmo], col-major
        mo1: &[f64],        // analdrv CP-HF solution [nmo*nocc], col-major
        vxc_d1_mo: Option<&[f64]>, // frozen-density v_xc skeleton, MO [nmo*nmo]
        fxc_cache: Option<&crate::dft::response::FxcHessianCache>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let n2 = nao * nao;

        // first-order density and complete first-order Fock
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let mut c1 = vec![0.0f64; nao * nocc];
        for i in 0..nocc {
            for p in 0..nmo {
                let m1 = mo1[p + i * nmo];
                if m1 != 0.0 {
                    for mu in 0..nao {
                        c1[mu + i * nao] += self.c_mo[[mu, p]] * m1;
                    }
                }
            }
        }
        let mut dm1 = vec![0.0f64; n2];
        for mu in 0..nao {
            for nu in 0..nao {
                let mut acc = 0.0;
                for i in 0..nocc {
                    acc += c1[mu + i * nao] * cocc[nu + i * nao]
                        + cocc[mu + i * nao] * c1[nu + i * nao];
                }
                dm1[mu + nu * nao] = 2.0 * acc;
            }
        }
        let dm1_m = to_mat(&dm1, nao, nao);
        let t0 = Instant::now();
        let j_up = crate::dft::response::compute_j_upper(self.scf, &vec![dm1_m.clone()]);
        timer_add(2, t0);
        let t0 = Instant::now();
        let k_up = crate::dft::response::compute_k_upper(self.scf, &vec![dm1_m.clone()]);
        timer_add(3, t0);
        let j_full = j_up.to_matrixfull().expect("gw_grad: J upper").data;
        let k_full = k_up.to_matrixfull().expect("gw_grad: K upper").data;
        // XC-kernel response of the density change (DFT references only)
        let fxc_ao: Option<MatrixFull<f64>> = fxc_cache
            .map(|c| crate::dft::response::compute_fxc_response_ao_cached(c, &dm1_m));

        // relaxed part of the first-order Fock in the AO basis ...
        let mut f1_relaxed = vec![0.0f64; n2];
        for k in 0..n2 {
            f1_relaxed[k] = j_full[k] - 0.5 * self.fock_hyb * k_full[k];
        }
        if let Some(fa) = fxc_ao.as_ref() {
            for (f, v) in f1_relaxed.iter_mut().zip(fa.data.iter()) {
                *f += v;
            }
        }
        // ... transformed to MO and added to the skeleton derivative
        let mut f1_mo: Vec<f64> = {
            let relaxed = {
                let t = gemm_nt(&self.c_mo.data, nao, nmo, &f1_relaxed, nao, nao);
                gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
            };
            f1_skel_mo
                .iter()
                .zip(relaxed.iter())
                .map(|(a, b)| a + b)
                .collect()
        };
        // Both derivative matrices are derivatives of symmetric matrices, so
        // the Roothaan restoration only ever reads their symmetric part.  Any
        // asymmetry is numerical (the triangular `to_matrixfull` mirrors, the
        // MO transformation preserves symmetry up to rounding); averaging it
        // away makes the orthogonality identity `u + u^T + S1 = 0` exact for
        // **every** pair, whatever its gap.  Without it the identity residual
        // of a symmetry-degenerate (or merely nearly degenerate) reference is
        // the rounding noise divided by `e_q - e_p`, which can abort a valid
        // calculation through the diagnostic below.
        // how asymmetric the raw derivatives are (the identity residual of a
        // near-degenerate pair is this noise divided by the gap)
        let (mut f1_asym, mut s1_asym) = (0.0f64, 0.0f64);
        for p in 0..nmo {
            for q in 0..nmo {
                f1_asym = f1_asym.max((f1_mo[p + q * nmo] - f1_mo[q + p * nmo]).abs());
                s1_asym = s1_asym.max((s1_mo[p + q * nmo] - s1_mo[q + p * nmo]).abs());
            }
        }
        // `REST_GWGRAD_NO_SYMM` restores the historical behaviour (raw
        // derivatives) and exists only so the degeneracy diagnostic can be
        // reproduced in a test.
        if std::env::var("REST_GWGRAD_NO_SYMM").is_err() {
            symmetrise_square(&mut f1_mo, nmo);
        }
        let s1_mo: Vec<f64> = {
            let mut v = s1_mo.to_vec();
            if std::env::var("REST_GWGRAD_NO_SYMM").is_err() {
                symmetrise_square(&mut v, nmo);
            }
            v
        };
        if std::env::var("REST_GWGRAD_DBG").is_ok() {
            println!("[cphf DBG] raw f1/s1 asymmetry {f1_asym:.3e} / {s1_asym:.3e}");
        }

        let mut u = vec![0.0f64; nmo * nmo];
        let mut eps1 = vec![0.0f64; nmo];
        for p in 0..nmo {
            u[p + p * nmo] = -0.5 * s1_mo[p + p * nmo];
            eps1[p] = f1_mo[p + p * nmo] - self.e[p] * s1_mo[p + p * nmo];
        }
        // Non-degenerate pairs: the usual Roothaan restoration.  Pairs inside
        // one degenerate shell are excluded (their `e_q - e_p` vanishes) and
        // handled below.
        for p in 0..nmo {
            for q in 0..nmo {
                if p == q || self.same_degenerate_shell(p, q) {
                    continue;
                }
                u[p + q * nmo] = (f1_mo[p + q * nmo] - self.e[q] * s1_mo[p + q * nmo])
                    / (self.e[q] - self.e[p]);
            }
        }
        // Degenerate shells: the perturbation is not diagonal in the SCF
        // basis, so its shell block
        //     A[i,j] = F1[p_i, p_j] - e * S1[p_i, p_j]      (e = shell energy)
        // is diagonalised; its eigenvalues are the level derivatives
        // `eps1[p_i]` and its eigenvectors are the first-order rotation of the
        // shell basis (degenerate perturbation theory).  `tr A` — and hence
        // the shell-averaged gradient — is basis independent either way, but
        // the rotation makes each member's `eps1` the derivative of its own
        // split level instead of an arbitrary mixture of the two.
        for shell in self.shells.iter() {
            let d = shell.len();
            if d < 2 {
                continue;
            }
            let e0 = shell.iter().map(|&p| self.e[p]).sum::<f64>() / d as f64;
            let mut a = vec![0.0f64; d * d];
            for (ii, &p) in shell.iter().enumerate() {
                for (jj, &q) in shell.iter().enumerate() {
                    a[ii + jj * d] = f1_mo[p + q * nmo] - e0 * s1_mo[p + q * nmo];
                }
            }
            for ii in 0..d {
                for jj in (ii + 1)..d {
                    let s = 0.5 * (a[ii + jj * d] + a[jj + ii * d]);
                    a[ii + jj * d] = s;
                    a[jj + ii * d] = s;
                }
            }
            let (lam, v) = eigh_symmetric(&a, d);
            for (ii, &p) in shell.iter().enumerate() {
                eps1[p] = lam[ii];
            }
            // antisymmetric part of V (the pure first-order rotation)
            for ii in 0..d {
                for jj in 0..d {
                    if ii != jj {
                        let (p, q) = (shell[ii], shell[jj]);
                        u[p + q * nmo] = 0.5 * (v[ii + jj * d] - v[jj + ii * d]);
                    }
                }
            }
        }
        // The orthogonality identity `u + u^T + S1 = 0` is imposed by the
        // non-degenerate formula; inside a shell a rotation of the basis is a
        // free gauge (the shell functional is invariant under it), so those
        // pairs are excluded from the diagnostic.  With the symmetrised
        // `f1_mo`/`s1_mo` above the identity is algebraically exact, so what is
        // measured here is the solve residual only (diffuse bases amplify it by
        // the smallest orbital gap).
        let mut orth = 0.0f64;
        let mut worst = (0usize, 0usize, 0.0f64);
        for p in 0..nmo {
            for q in 0..nmo {
                if self.same_degenerate_shell(p, q) {
                    continue;
                }
                let e = (u[p + q * nmo] + u[q + p * nmo] + s1_mo[p + q * nmo]).abs();
                if e > worst.2 {
                    worst = (p, q, e);
                }
                orth = orth.max(e);
            }
        }
        if orth > 1.0e-2 {
            panic!(
                "gw_grad: canonical response orthogonality error {:.3e} at pair ({}, {}) \
                 (gap {:.3e}, degenerate_tol {:.1e})",
                orth,
                worst.0,
                worst.1,
                (self.e[worst.1] - self.e[worst.0]).abs(),
                self.config.degenerate_tol
            );
        }
        if orth > 1.0e-6 {
            println!(
                "warning!!! gw_grad canonical response orthogonality error {:.3e} at pair ({}, {})",
                orth, worst.0, worst.1
            );
        }
        // ---- v_xc part of the static-term covector b_x (DFT references) ----
        // `b_x[p] = -d v_xc,pp / dR`: explicit grid derivative (frozen density
        // and MOs), orbital rotation `2 sum_q U[q,p] v_xc[q,p]` and the
        // XC-kernel response `<p| fxc(D^x) |p>` of the first-order density.
        // The `(1-alpha) Sigma_x` half of `B_n` needs nothing here — its
        // complete derivative (explicit + orbital) flows through the
        // (bq, bJ) exchange pullback of `qp_pullback_chunk`.
        let mut b_x = vec![0.0f64; nmo];
        if let Some(vd1) = vxc_d1_mo {
            // MO diagonal of the fxc response: diag(C^T fxc_ao C)
            let fxc_diag: Vec<f64> = if let Some(fa) = fxc_ao.as_ref() {
                let t = gemm_nt(&self.c_mo.data, nao, nmo, &fa.data, nao, nao);
                (0..nmo)
                    .map(|p| {
                        let base = p * nao;
                        (0..nao)
                            .map(|mu| t[p + mu * nmo] * self.c_mo.data[mu + base])
                            .sum::<f64>()
                    })
                    .collect()
            } else {
                vec![0.0f64; nmo]
            };
            for p in 0..nmo {
                let mut s = vd1[p + p * nmo] + fxc_diag[p];
                let col = p * nmo;
                for q in 0..nmo {
                    s += 2.0 * u[q + col] * self.vxc_mo[q + col];
                }
                b_x[p] = -s;
            }
        }
        (u, eps1, b_x)
    }

    /// Analytic gradient of one AC / WAC G0W0 QP energy.
    ///
    /// The LR-CD derivative path has been removed; CD is differentiated by
    /// [`GwCdGradEngine`].  This driver only builds the low-rank AC/WAC QP
    /// caches and contracts their pullback.
    ///
    /// Returns `(grad, omega, z_factor)` with `grad[atm*3 + comp]` in
    /// Hartree/Bohr (row-major over atoms and Cartesian components).
    ///
    /// When `target` belongs to a symmetry-protected degenerate shell the
    /// returned gradient is the **shell average** `(1/d) sum_m d Omega_m/dR`
    /// and `omega` is the QP energy of `target` itself (all shell members are
    /// degenerate).  This is the only basis-independent definition of "the"
    /// gradient there, and it is what the central finite difference of the
    /// QP energy of orbital `target` converges to: displacing the nuclei
    /// splits the shell, the sorted index follows the *other* branch at `-h`,
    /// so the two-sided difference averages the two level derivatives.
    pub fn analytic_gradient_ac_wac(&self, target: usize) -> (Vec<f64>, f64, f64) {
        assert!(
            matches!(self.config.gw_variant, GwVariant::Ac | GwVariant::Wac),
            "GwGradEngine::analytic_gradient_ac_wac covers only AC/WAC; use GwCdGradEngine for CD"
        );
        mem_probe("gradient: enter");
        let shell = self.degenerate_shell(target);
        let caches: Vec<QpCache> = shell.iter().map(|&m| self.build_target_cache(m)).collect();
        let w = 1.0 / caches.len() as f64;
        let refs: Vec<&QpCache> = caches.iter().collect();
        let adjoint_route = self.config.adjoint_cphf && self.supports_cphf_adjoint();
        // `qt`/`qtia` are read only while building a QP cache (`consts`, the exact-residue
        // on-the-fly factors); the gradient's own contractions never touch them. Release
        // `naux*nmo^2 + naux*nov` doubles (949 MB at naphthalene/def2-TZVP) for the rest of the
        // gradient; `with_qt`/`with_qtia` rebuild them on demand.
        self.evict_qt();
        // On the HF-only adjoint route the pullback never reads `q` (its only `q`-dependent term,
        // the static exchange derivative, is skipped for `fock_hyb == 1`) and the fold below uses
        // the streamed-integrals variant, so `q` (`naux*nmo^2` doubles, 874 MB at
        // naphthalene/def2-TZVP) can be released *before* the pullback instead of after the fold.
        // Any later reader rebuilds it through `with_q`.
        if adjoint_route {
            self.evict_q();
        }
        // The adjoint route rebuilds the raw integrals only inside `build_cphf_adjoint`, after the
        // response objects (and their kernel cache) have been released; keeping them out of the
        // covector/fold phase removes `nao^2 * naux` doubles from where this gradient peaks.
        // (The pullback and the fold read `q`/`qtia`/factors, never `i3`.)
        if adjoint_route {
            self.raw.evict_i3();
        }
        mem_probe("gradient: after QP caches");
        let (mut bq, bj, be, bb) = self.qp_pullback(&refs, &[vec![w; caches.len()]]);
        let own0 = &caches[shell.iter().position(|&m| m == target).unwrap()];
        mem_probe("gradient: after qp_pullback (bq/bj/be/bb)");
        // The pullback's parallel regions each allocate and free their own block
        // scratch; glibc keeps those blocks in the *per-thread* arenas, so the
        // resident set after the pullback is ~175 MB higher with eight workers than
        // with one even though the live data is identical.  The fold, which is where
        // this gradient peaks, would then start from that inflated baseline.
        // Returning the free arenas to the OS here keeps the eight-thread curve on
        // top of the one-thread one.
        crate::hessian::memory_monitor::trim_to_os(0);
        // ---- adjoint (Z-vector) route --------------------------------
        // The reference guard is checked *before* folding (both the builder and
        // this branch use `supports_cphf_adjoint`): the folded covector is all
        // the adjoint route needs from `bq`, so the `naux * nmo^2` covector
        // buffer (874 MB at naphthalene/def2-TZVP) is released before the
        // adjoint workspace is built. The adjoint foldings and the
        // per-direction loop both then run without it resident. The
        // per-perturbation fallback below still needs `bq`, which is why the
        // release happens only on this branch.
        if adjoint_route {
            // `bq` was symmetrised by the pullback, so compact it in place to its lower triangle
            // before folding (the fold rebuilds the symmetric plane exactly): this releases
            // `naux*nmo^2/2` doubles (437 MB at naphthalene/def2-TZVP) for the fold, which is where
            // this gradient peaks. The route always returns below, so the (full-buffer) fallback
            // never sees the compacted form.
            // the pullback accumulates the pair covector directly in its packed
            // (lower-triangle) layout, so no repacking is needed here
            let packed_pairs = packed_pair_count(self.nmo);
            // no `q` here: the orbital-rotation channel is contracted against the raw integrals,
            // which are streamed auxiliary-atom block by block
            let mut fold =
                self.fold_covector_packed_streamed_parallel(&bq[0][..packed_pairs * self.naux]);
            // the folded covector replaces `bq` entirely, so release it here (the route returns
            // below; the full-buffer fallback never sees the compacted form)
            drop(bq);
            mem_probe("gradient: after fold_covector");
            // `q` is read only by the pullback and this fold, both done: release `naux*nmo^2`
            // doubles (874 MB at naphthalene/def2-TZVP) before the adjoint workspace is built.
            // `with_q` rebuilds it if a later task (another target, a BSE screening) needs it.
            self.evict_q();
            // `build_cphf_adjoint` takes `b_ao` out of `fold` and folds the exchange skeleton into
            // it in place, so its `naux * nao * nao` tensor becomes `adj.di_cov` instead of being
            // duplicated; only the small `w_rot` is left to release here.
            let adj = self
                .build_cphf_adjoint(&mut fold, &be[0], Some(&bb[0]))
                .expect("`supports_cphf_adjoint` and `build_cphf_adjoint` share this guard");
            drop(fold);
            mem_probe("gradient: after build_cphf_adjoint");
            let mut grad = vec![0.0f64; self.natm * 3];
            // Explicit v_xc grid derivative of a DFT reference (one sweep, shared
            // by every perturbation); `None` for HF, where `b_x` is identically 0.
            let vxc_atoms = self.vxc_grid_derivative_atoms();
            for atm in 0..self.natm {
                let h1r = crate::hessian::rhf::build_hcore_first_deriv(&self.scf.mol, atm);
                let s1r = crate::ri_cphf::cphf_solver_pyscf::build_s1ao_deriv(&self.scf.mol, atm);
                // the locality-aware variant streams the derivative blocks internally, so it does
                // not need (or allocate) the dense per-atom `AtomDerivBlocks`
                if atm == 0 {
                    mem_probe("gradient: inside per-direction loop (1st dir)");
                }
                let bx3: Vec<Vec<f64>> = match vxc_atoms.as_ref() {
                    Some(a) => (0..3).map(|c| self.vxc_diag_bx(a, atm, c)).collect(),
                    None => vec![vec![0.0f64; self.nmo]; 3],
                };
                if self.config.local_di {
                    let djs: Vec<Vec<f64>> = (0..3).map(|c| self.raw.d_j_atom(atm, c)).collect();
                    let g3 = self.adjoint_perturbation_gradient_local(
                        &adj, atm, &h1r, &s1r, &djs, &bj[0], &bb[0], &bx3,
                        vxc_atoms.as_ref().map(|a| &a[atm]), atm * 3,
                    );
                    grad[atm * 3..atm * 3 + 3].copy_from_slice(&g3);
                } else {
                    let blocks = self.raw.d_atom_blocks(atm);
                    for comp in 0..3 {
                        let dj = self.raw.d_j_atom(atm, comp);
                        grad[atm * 3 + comp] = self.adjoint_perturbation_gradient(
                            &adj, &blocks, comp, &h1r, &s1r[comp], &dj, &bj[0], &bb[0],
                            &bx3[comp], vxc_atoms.as_ref().map(|a| &a[atm]), atm * 3 + comp,
                        );
                    }
                }
            }
            mem_probe("gradient: after per-direction loop (adjoint)");
            return (grad, own0.omega, own0.z_factor);
        }
        // responses before the pullback: see analytic_gradient_with_cache
        // for the memory-peak rationale of this order
        let responses = self.canonical_response_batch();
        // Fold the (single) weight row's `bq` once: the AO-basis covector
        // `B = C bq C^T` and the rotation factor `W = sum_P q_P bq_P` replace
        // the per-perturbation `qx` assembly (see `FoldedCovector`).
        let folded = if self.config.folded_qx {
            Some(self.fold_covector_packed(&bq[0]))
        } else {
            None
        };
        // the legacy per-perturbation route contracts the **full** pair layout
        let bq_full: Vec<Vec<f64>> = if self.config.folded_qx {
            Vec::new()
        } else {
            bq.iter().map(|v| unpack_pairs(v, self.nmo, self.naux)).collect()
        };
        let mut grad = vec![0.0f64; self.natm * 3];
        for atm in 0..self.natm {
            let blocks = self.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let dj = self.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let g = match &folded {
                    Some(f) => self.folded_perturbation_gradient(
                        f, &blocks, comp, &u, &bj[0], &be[0], &bb[0], &dj, &eps1, &b_x,
                    ),
                    None => {
                        let qx = self.qx_from_u_blocks(&blocks, comp, &u);
                        self.contract_perturbation(
                            &bq_full, &bj, &be, &bb, &qx, &dj, &eps1, &b_x,
                        )[0]
                    }
                };
                grad[atm * 3 + comp] = g;
            }
        }
        let own = &caches[shell.iter().position(|&m| m == target).unwrap()];
        (grad, own.omega, own.z_factor)
    }

    /// QP energy of one orbital (the scalar whose gradient is returned by
    /// [`GwGradEngine::analytic_gradient_ac_wac`]).
    pub fn qp_energy_of(&self, target: usize) -> f64 {
        self.build_target_cache(target).omega
    }

}

// ---------------------------------------------------------------------------
// free helpers
