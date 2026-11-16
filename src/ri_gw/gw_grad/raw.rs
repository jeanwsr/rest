
/// Geometry-fixed raw RI data: `(mu nu | P)`, `J = (P|Q)` and the machinery
/// needed to assemble any nuclear derivative `dI`, `dJ`.
///
/// The electronic derivative tensors `dI1`/`dI2` are NOT stored (they would
/// cost 2 x 3 nao^2 naux doubles, several GB at naphthalene scale).
/// Instead, [`RawRiTensors::d_atom_blocks`] evaluates the derivative
/// integrals shell-sliced on the perturbed atom's centers on the fly —
/// the total integral work is identical to one full `dI1`/`dI2` build, but
/// only one atom's blocks (O(nao_A nao naux)) are resident at a time.
pub struct RawRiTensors {
    pub nao: usize,
    pub naux: usize,
    pub natm: usize,
    /// I[mu + nu*nao + P*nao*nao], col-major [nao, nao, naux].
    ///
    /// Held as `Arc` inside a `Mutex<Option<..>>` rather than in a [`RefCell`], so
    /// that the many rayon workers of the canonical CP-KS assembly can read it
    /// concurrently while one of them is inside a long [`RawRiTensors::with_i3`]
    /// closure (`j_deriv_from` / `k_deriv_from` / `build_q` all wrap their whole
    /// body in one).  With a `RefCell` those readers could neither borrow nor make
    /// progress: `ensure_i3` returned immediately while the cell was borrowed, so
    /// `with_i3` **spun at 100% CPU** until the holder finished.  Measured on
    /// acetylene/def2-TZVPP: three workers spinning for hours, ~44k wasted
    /// CPU-seconds on a job whose real work is ~12.7k, i.e. a phase that runs
    /// single-threaded while looking busy.
    ///
    /// Now a reader clones the `Arc` and drops the lock at once (readers never
    /// block each other, nothing spins), and the build happens under the lock, so
    /// a concurrent caller sleeps on the mutex instead of recomputing the
    /// integrals.  `nao^2 * naux` doubles (874 MB at naphthalene/def2-TZVP) are
    /// still released by [`RawRiTensors::evict_i3`], which drops the `Arc`.
    pub i3: Mutex<Option<Arc<Vec<f64>>>>,
    /// J[P + Q*naux]
    pub j2: Vec<f64>,
    /// dJ1[P + Q*naux + t*naux*naux]
    pub dj1: Vec<f64>,
    /// AO index -> atom
    pub ao_atom: Vec<usize>,
    /// auxiliary AO index -> atom
    pub aux_atom: Vec<usize>,
    /// integral engines for the on-the-fly derivative assembly
    pub mol: CInt,
    pub auxmol: CInt,
    /// per-atom [shl0, shl1] of the orbital basis
    pub ao_shl_range: Vec<[usize; 2]>,
    /// per-atom [ao0, ao1] of the orbital basis
    pub ao_range: Vec<[usize; 2]>,
    /// per-atom [shl0, shl1] of the auxiliary basis
    pub aux_shl_range: Vec<[usize; 2]>,
    /// per-atom [aux0, aux1]
    pub aux_range: Vec<[usize; 2]>,
}

/// Shell-sliced derivative integral blocks of ONE perturbed atom, shared by
/// its three Cartesian components (PySCF `gw_cd_grad_optimized.
/// _AtomThreeCenterDerivatives`).
pub struct AtomDerivBlocks {
    /// dI1 slice: [nrow, nao, naux, 3] col-major, derivative on the atom's
    /// own (bra) centers
    pub d1: Vec<f64>,
    /// dI2 slice: [nao, nao, naux_a, 3] col-major, derivative on the atom's
    /// auxiliary centers
    pub d2: Vec<f64>,
    /// atom index
    pub atm: usize,
    /// first AO of the atom
    pub row0: usize,
    /// number of AOs of the atom
    pub nrow: usize,
    /// number of auxiliary functions of the atom
    pub naux_a: usize,
    /// first auxiliary function of the atom
    pub aux0: usize,
}

fn atom_map_from_slices(slices: &[[usize; 4]], nao: usize) -> Vec<usize> {
    let mut map = vec![0usize; nao];
    for (atm, s) in slices.iter().enumerate() {
        for p in s[2] as usize..s[3] as usize {
            map[p] = atm;
        }
    }
    map
}

/// Build the raw RI tensors with the libcint electronic-derivative integrals.
pub fn build_raw_ri_tensors(scf: &SCF) -> RawRiTensors {
    let mol = get_cint_mol(&scf.mol);
    let aux = get_cint_aux(&scf.mol);
    let nao = mol.nao();
    let naux = aux.nao();
    let natm = scf.mol.geom.nfree;

    // The three-centre integrals are evaluated **on demand**: the engine build and the CD-GW
    // gradient consume them in auxiliary-atom blocks (`for_each_i3_atom_block`, so the dense
    // `nao^2 * naux` array — 874 MB at naphthalene/def2-TZVP — is never materialised), and only the
    // per-perturbation route and the TDDFT gradient read them densely (`with_i3`, which evaluates
    // them once and keeps them). Leaving this empty therefore costs those paths one integral
    // evaluation and keeps the block out of every other task's peak.

    let (j2, shapej) = aux.integrate("int2c2e", "s1", None).into();
    assert_eq!(shapej, vec![naux, naux], "int2c2e shape");
    let (dj1, shapedj) = aux.integrate("int2c2e_ip1", "s1", None).into();
    assert_eq!(shapedj, vec![naux, naux, 3], "int2c2e_ip1 shape");

    let ao_slices = scf.mol.aoslice_by_atom();
    let aux_slices = scf.mol.make_auxmol_fake().aoslice_by_atom();
    let range = |s: &[[usize; 4]], k0: usize, k1: usize| -> Vec<[usize; 2]> {
        s.iter().map(|r| [r[k0], r[k1]]).collect()
    };

    RawRiTensors {
        nao,
        naux,
        natm,
        i3: Mutex::new(None),
        j2,
        dj1,
        ao_atom: atom_map_from_slices(&ao_slices, nao),
        aux_atom: atom_map_from_slices(&aux_slices, naux),
        mol,
        auxmol: aux,
        ao_shl_range: range(&ao_slices, 0, 1),
        ao_range: range(&ao_slices, 2, 3),
        aux_shl_range: range(&aux_slices, 0, 1),
        aux_range: range(&aux_slices, 2, 3),
    }
}

impl RawRiTensors {
    /// Run `f` on the raw three-centre integrals, re-evaluating them if a previous
    /// [`Self::evict_i3`] released them.
    ///
    /// The closure runs on an owned `Arc` handle, *not* while a lock is held: several rayon
    /// workers may be inside `with_i3` at the same time, and one of them may run for a long
    /// time (the canonical `j_deriv_from`/`k_deriv_from` wrap their whole body in it).  With
    /// the previous `RefCell` implementation those concurrent readers could neither borrow
    /// nor make progress -- `ensure_i3` returned immediately while the cell was borrowed and
    /// the retry loop spun at 100% CPU.  Nothing inside may call [`Self::evict_i3`] and
    /// expect the release to take effect immediately: the caller's `Arc` keeps the buffer
    /// alive until it returns.
    pub fn with_i3<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        let i3 = self.i3_shared();
        f(i3.as_slice())
    }

    /// The resident three-centre integrals, built on first use.
    ///
    /// Building happens under the lock, so concurrent callers wait (sleeping on the mutex)
    /// instead of each re-evaluating `int3c2e` and throwing all but one result away.
    pub fn i3_shared(&self) -> Arc<Vec<f64>> {
        let mut slot = self.i3.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(resident) = slot.as_ref() {
            return Arc::clone(resident);
        }
        let t0 = Instant::now();
        let (i3, shape3) =
            CInt::integrate_cross("int3c2e", [&self.mol, &self.mol, &self.auxmol], "s1", None).into();
        if std::env::var("REST_GWGRAD_MEMDBG").is_ok() {
            println!(
                "[memdbg]   i3 re-evaluation (int3c2e, {:.1} MB): {:.3}s",
                (self.nao * self.nao * self.naux) as f64 * 8.0 / 1048576.0,
                t0.elapsed().as_secs_f64()
            );
        }
        assert_eq!(shape3, vec![self.nao, self.nao, self.naux], "int3c2e s1 shape");
        let arc = Arc::new(i3);
        *slot = Some(Arc::clone(&arc));
        arc
    }

    /// Rebuild the three-centre integrals when they are not resident.
    ///
    /// Callers that need the buffer for a whole routine (the TDDFT gradient's Coulomb/`Y`
    /// helpers) take it once with [`Self::i3_shared`] and keep the `Arc`.
    pub fn ensure_i3(&self) {
        let _ = self.i3_shared();
    }

    /// Evaluate the raw three-centre integrals **auxiliary-atom block by auxiliary-atom block** and
    /// hand each block to `f` as `(aux_offset, naux_block, block)`, with `block` col-major
    /// `[nao, nao, naux_block]` (auxiliary functions ordered as in the full tensor, i.e. starting at
    /// `aux_offset`).
    ///
    /// This is the streaming alternative to [`Self::with_i3`] for callers that only *contract* the
    /// integrals: the full `nao^2 * naux` array (874 MB at naphthalene/def2-TZVP) is never
    /// materialised, at the price of evaluating the same shell triplets in a few more libcint calls
    /// (the per-call shell-pair setup is negligible against the integral work). The integral count
    /// is identical to one full `int3c2e` call.
    pub fn for_each_i3_atom_block<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let nao = self.nao;
        for atm in 0..self.natm {
            let fun = self.aux_range[atm];
            let nb = fun[1] - fun[0];
            if nb == 0 {
                continue;
            }
            let shls = [[0, self.mol.nbas()], [0, self.mol.nbas()], self.aux_shl_range[atm]];
            let (blk, shape) =
                CInt::integrate_cross("int3c2e", [&self.mol, &self.mol, &self.auxmol], "s1", &shls).into();
            assert_eq!(shape, vec![nao, nao, nb], "auxiliary-atom int3c2e block shape");
            f(fun[0], nb, &blk);
        }
    }

    /// Stream the **AO-centre derivative** blocks (`int3c2e_ip1`) of one perturbed atom over
    /// auxiliary-shell blocks, handing `f(aux_offset, naux_block, block)` with `block` col-major
    /// `[nrow_A, nao, naux_block, 3]`.
    ///
    /// The per-direction contraction only *reads* these blocks, so streaming them keeps
    /// `nrow_A * nao * naux * 3` doubles out of the gradient's peak (254 MB for a carbon atom at
    /// naphthalene/def2-TZVP, against a few MB per block). The integral work is the same
    /// shell-triplet sum as one `int3c2e_ip1` call, split into a few more libcint calls.
    pub fn for_each_d1_block<F>(&self, atm: usize, mut f: F)
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let row = self.ao_range[atm];
        let nrow = row[1] - row[0];
        let nshl_aux = self.auxmol.nbas();
        // a few dozen auxiliary shells per block keeps the block (~60 MB) far below the full tensor
        // (254 MB) while keeping the number of libcint calls small
        let step = 64usize;
        let mut a0 = 0usize;
        let mut fun_off = 0usize;
        while a0 < nshl_aux {
            let a1 = (a0 + step).min(nshl_aux);
            let shls = [self.ao_shl_range[atm], [0, self.mol.nbas()], [a0, a1]];
            let (blk, shape) =
                CInt::integrate_cross("int3c2e_ip1", [&self.mol, &self.mol, &self.auxmol], "s1", &shls).into();
            let nb = shape[2];
            assert_eq!(shape, vec![nrow, self.nao, nb, 3], "streamed int3c2e_ip1 block shape");
            f(fun_off, nb, &blk);
            fun_off += nb;
            a0 = a1;
        }
        debug_assert_eq!(fun_off, self.naux);
    }

    /// Stream the **auxiliary-centre derivative** blocks (`int3c2e_ip2`) of one perturbed atom over
    /// its own auxiliary shells; `block` is col-major `[nao, nao, naux_block, 3]`.
    pub fn for_each_d2_block<F>(&self, atm: usize, mut f: F)
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let (shl0, shl1) = (self.aux_shl_range[atm][0], self.aux_shl_range[atm][1]);
        let fun0 = self.aux_range[atm][0];
        // the atom's own auxiliary shells are only a handful, so small blocks are cheap here and
        // cut the `nao^2 * naux_A * 3` tensor (135 MB for a carbon atom) to a few tens of MB
        let step = 4usize;
        let mut a0 = shl0;
        let mut fun_off = 0usize;
        while a0 < shl1 {
            let a1 = (a0 + step).min(shl1);
            let shls = [[0, self.mol.nbas()], [0, self.mol.nbas()], [a0, a1]];
            let (blk, shape) =
                CInt::integrate_cross("int3c2e_ip2", [&self.mol, &self.mol, &self.auxmol], "s1", &shls).into();
            let nb = shape[2];
            assert_eq!(shape, vec![self.nao, self.nao, nb, 3], "streamed int3c2e_ip2 block shape");
            f(fun0 + fun_off, nb, &blk);
            fun_off += nb;
            a0 = a1;
        }
    }

    /// The deterministic work batches (auxiliary-function ranges) used by the parallel
    /// perturbation contractions.
    ///
    /// The batches tile the **whole** auxiliary basis, not only the perturbed atom's own
    /// functions: `int3c2e_ip1` differentiates `(mu_A nu | P)` with respect to the position of
    /// atom `A`, and the atom's `mu` rows couple to *every* auxiliary function `P`.  (The
    /// `int3c2e_ip2` part is the mirror image — only the atom's own `P` carry the auxiliary-centre
    /// derivative — and the batch contraction accounts for that by intersecting each batch with
    /// `aux_range[atm]`, an intersection that is empty for most batches.)  Getting this range
    /// wrong is not a small effect: it silently drops the `ip1` contribution of every auxiliary
    /// function not centred on the perturbed atom.
    ///
    /// The layout is a function of the auxiliary shell indices alone — never of the thread
    /// count — so the callers can process the batches in parallel and merge their partial sums in
    /// index order and stay bit-for-bit reproducible under `num_threads`, the same discipline as
    /// [`crate::utilities::par_sum`].  Each batch spans at least [`PERT_BLK`] auxiliary functions
    /// (one whole shell more at most), which keeps the per-batch integral calls few and their
    /// result small: `nrow_A * nao * nb * 3`, i.e. a few MB at `PERT_BLK`, against the
    /// `nrow_A * nao * naux * 3` of the full per-atom tensor (254 MB for a carbon atom at
    /// naphthalene/def2-TZVP).  Every batch is a whole number of auxiliary shells, so the `ip1`
    /// block it evaluates is exactly the slice it contracts, and the batches together evaluate
    /// each `ip1` shell triplet exactly once.
    pub fn aux_perturbation_batches(&self) -> Vec<[usize; 2]> {
        Self::perturbation_batches(self.auxmol.nbas(), &self.auxmol.ao_loc())
    }

    /// Pure-function core of [`Self::aux_perturbation_batches`]: everything the batch layout
    /// depends on, so that callers can compute it once and then run the batches in parallel
    /// without holding a `&self` (`RawRiTensors` is not `Sync` because of its `RefCell`
    /// caches).
    fn perturbation_batches(nshl: usize, loc: &[usize]) -> Vec<[usize; 2]> {
        let mut out: Vec<[usize; 2]> = Vec::new();
        let mut a0 = 0usize;
        while a0 < nshl {
            let f0 = loc[a0];
            let mut a1 = a0;
            let mut f1 = f0;
            while a1 < nshl && f1 - f0 < PERT_BLK {
                f1 = loc[a1 + 1];
                a1 += 1;
            }
            if f1 > f0 {
                out.push([f0, f1]);
            }
            a0 = a1;
        }
        debug_assert!(out.is_empty() || out[0][0] == 0);
        debug_assert_eq!(out.last().map(|r| r[1]), loc.get(nshl).copied());
        out
    }

    /// Pure-function core of the per-batch block materialisation, so that the perturbation
    /// contraction can run its batches in parallel from `&CInt`s alone (`RawRiTensors` itself is
    /// not `Sync` because of its `RefCell` caches).
    ///
    /// `[f0, f1)` must be a whole number of auxiliary shells, which is what
    /// [`Self::aux_perturbation_batches`] produces; `nrow` is the number of AO functions of the
    /// perturbed atom (`ao_range[atm][1] - ao_range[atm][0]`).
    ///
    /// Returns `(d1, d2)`, both **in the layouts of [`Self::for_each_d1_block`] /
    /// [`Self::for_each_d2_block`]**, i.e. col-major `[nrow, nao, f1 - f0, 3]` and col-major
    /// `[nao, nao, naux_a, 3]` with `naux_a` the number of the atom's own auxiliary functions
    /// inside the batch (zero for every batch that misses them).  Because the batch runs over its
    /// own shells, the `ip1` integral *is* the block — no slicing or re-layout is needed, and the
    /// batches together cost one full `int3c2e_ip1` pass, not one per batch.
    pub fn perturbation_batch_blocks_pure(
        mol: &CInt,
        auxmol: &CInt,
        ao_shl_range: &[usize; 2],
        aux_range: &[usize; 2],
        nrow: usize,
        nao: usize,
        f0: usize,
        f1: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        // `ip1` is evaluated over the batch's auxiliary shells and covers the atom's `mu` rows
        // against every `nu`; `ip2` is dense in `[mu,nu]` but only for the atom's own auxiliary
        // functions, so its integral range is the batch's intersection with them.
        let loc = auxmol.ao_loc();
        let nb = f1 - f0;
        let mut d1 = Vec::new();
        if nb > 0 {
            let s_lo = loc.partition_point(|&f| f < f0);
            let s_hi = loc.partition_point(|&f| f < f1);
            assert_eq!(
                (loc[s_lo], loc[s_hi]),
                (f0, f1),
                "perturbation batch must span whole auxiliary shells"
            );
            let (d, shape1) = CInt::integrate_cross(
                "int3c2e_ip1",
                [mol, mol, auxmol],
                "s1",
                &[*ao_shl_range, [0, mol.nbas()], [s_lo, s_hi]],
            )
            .into();
            assert_eq!(shape1, vec![nrow, nao, nb, 3], "int3c2e_ip1 shape");
            d1 = d;
        }
        let n2 = nao * nao;
        let (a_lo, a_hi) = (f0.max(aux_range[0]), f1.min(aux_range[1]));
        let naux_a = a_hi.saturating_sub(a_lo);
        let mut d2c = vec![0.0f64; n2 * naux_a * 3];
        if naux_a > 0 {
            let c_lo = loc.partition_point(|&f| f < a_lo);
            let c_hi = loc.partition_point(|&f| f < a_hi);
            let (d2, shape2) = CInt::integrate_cross(
                "int3c2e_ip2",
                [mol, mol, auxmol],
                "s1",
                &[[0, mol.nbas()], [0, mol.nbas()], [c_lo, c_hi]],
            )
            .into();
            assert_eq!(shape2, vec![nao, nao, naux_a, 3], "int3c2e_ip2 shape");
            d2c.copy_from_slice(&d2);
        }
        (d1, d2c)
    }

    /// `int3c2e(mu,nu,P)` for one atom's `mu` rows over the auxiliary shell range `shl_range`,
    /// as `(values, nfun_lo, nfun_hi)` with `values` col-major `[nrow, nao, nfun]`.
    ///
    /// The pure (batchable) core of [`Self::for_each_i3_atom_block`], so a fold that only needs
    /// part of the auxiliary basis can run its parts in parallel without materialising the full
    /// `nao^2 * naux` tensor.
    pub fn i3_atom_block_pure(
        mol: &CInt,
        auxmol: &CInt,
        ao_shl_range: &[usize; 2],
        nrow: usize,
        nao: usize,
        shl_range: &[usize; 2],
    ) -> (Vec<f64>, usize, usize) {
        let loc = auxmol.ao_loc();
        let (f0, f1) = (loc[shl_range[0]], loc[shl_range[1]]);
        if f1 == f0 {
            return (Vec::new(), f0, f1);
        }
        let (blk, shape) = CInt::integrate_cross(
            "int3c2e",
            [mol, mol, auxmol],
            "s1",
            &[*ao_shl_range, [0, mol.nbas()], *shl_range],
        )
        .into();
        assert_eq!(shape, vec![nrow, nao, f1 - f0], "int3c2e block shape");
        (blk, f0, f1)
    }

    /// Length of the resident raw-integral buffer (0 while evicted) -- profiling aid.
    pub fn i3_len(&self) -> usize {
        self.i3
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or(0, |v| v.len())
    }

    /// Release the raw three-centre integrals (`nao^2 * naux` doubles).
    ///
    /// They are re-evaluated on demand by [`Self::with_i3`], so this only trades memory for the
    /// integral work of whoever needs them next. Tasks that hold their own copies of what they
    /// need (the CD-GW adjoint gradient after its Coulomb/`Y` contractions) can drop them for
    /// the rest of the calculation.  Handles already taken with [`Self::i3_shared`] keep the
    /// buffer alive until they are dropped.
    pub fn evict_i3(&self) {
        *self.i3.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Shell-sliced derivative integral blocks of one perturbed atom,
    /// shared by its three Cartesian components.  The total integral work
    /// summed over atoms equals one full `dI1`/`dI2` build, but only one
    /// atom's blocks are resident at a time.
    pub fn d_atom_blocks(&self, atm: usize) -> AtomDerivBlocks {
        let row = self.ao_range[atm];
        let nrow = row[1] - row[0];
        let aux_r = self.aux_range[atm];
        let naux_a = aux_r[1] - aux_r[0];
        let shls1 = [self.ao_shl_range[atm], [0, self.mol.nbas()], [0, self.auxmol.nbas()]];
        let (d1, shape1) = CInt::integrate_cross(
            "int3c2e_ip1",
            [&self.mol, &self.mol, &self.auxmol],
            "s1",
            &shls1,
        )
        .into();
        assert_eq!(
            shape1,
            vec![nrow, self.nao, self.naux, 3],
            "sliced int3c2e_ip1 shape"
        );
        let shls2 = [[0, self.mol.nbas()], [0, self.mol.nbas()], self.aux_shl_range[atm]];
        let (d2, shape2) = CInt::integrate_cross(
            "int3c2e_ip2",
            [&self.mol, &self.mol, &self.auxmol],
            "s1",
            &shls2,
        )
        .into();
        assert_eq!(
            shape2,
            vec![self.nao, self.nao, naux_a, 3],
            "sliced int3c2e_ip2 shape"
        );
        AtomDerivBlocks { d1, d2, atm, row0: row[0], nrow, naux_a, aux0: aux_r[0] }
    }

    /// Total nuclear derivative `dI_{mu nu P}/dR_{atm,comp}` (PySCF
    /// `gw_cd_grad._dI_atom`) assembled from the atom's sliced blocks: the
    /// Gaussian nuclear derivative is minus the electronic derivative on
    /// the corresponding center.
    pub fn d_i_from_blocks(&self, blocks: &AtomDerivBlocks, comp: usize) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let nao2 = nao * nao;
        let nrow = blocks.nrow;
        let stride1 = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = nao2 * blocks.naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];
        let mut di = vec![0.0f64; nao2 * naux];
        for mu in blocks.row0..blocks.row0 + nrow {
            for nu in 0..nao {
                for p in 0..naux {
                    di[mu + nu * nao + p * nao2] -=
                        d1c[(mu - blocks.row0) + nu * nrow + p * nrow * nao];
                }
            }
        }
        for nu in blocks.row0..blocks.row0 + nrow {
            for mu in 0..nao {
                for p in 0..naux {
                    di[mu + nu * nao + p * nao2] -=
                        d1c[(nu - blocks.row0) + mu * nrow + p * nrow * nao];
                }
            }
        }
        for p_l in 0..blocks.naux_a {
            let p = blocks.aux0 + p_l;
            for k in 0..nao2 {
                di[p * nao2 + k] -= d2c[p_l * nao2 + k];
            }
        }
        di
    }

    /// Assemble the `[nao, nao]` slice of `dI` for ONE auxiliary function
    /// `p` into `out` (col-major, overwritten).  Streaming variant of
    /// [`RawRiTensors::d_i_from_blocks`]: lets the qx U-part-gemm consume
    /// `dI` block by block so the full `nao^2 naux` derivative never
    /// coexists with the full `qx`.
    pub fn d_i_block(
        &self,
        blocks: &AtomDerivBlocks,
        comp: usize,
        p: usize,
        out: &mut [f64],
    ) {
        let nao = self.nao;
        let nao2 = nao * nao;
        let nrow = blocks.nrow;
        let stride1 = nrow * nao * self.naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = nao2 * blocks.naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];
        out.fill(0.0);
        for mu in blocks.row0..blocks.row0 + nrow {
            for nu in 0..nao {
                out[mu + nu * nao] -= d1c[(mu - blocks.row0) + nu * nrow + p * nrow * nao];
            }
        }
        for nu in blocks.row0..blocks.row0 + nrow {
            for mu in 0..nao {
                out[mu + nu * nao] -= d1c[(nu - blocks.row0) + mu * nrow + p * nrow * nao];
            }
        }
        if p >= blocks.aux0 && p < blocks.aux0 + blocks.naux_a {
            let base = (p - blocks.aux0) * nao2;
            for k in 0..nao2 {
                out[k] -= d2c[base + k];
            }
        }
    }

    /// Total nuclear derivative `dJ_{PQ}/dR_{atm,comp}`.
    pub fn d_j_atom(&self, atm: usize, comp: usize) -> Vec<f64> {
        let naux = self.naux;
        let naux2 = naux * naux;
        let mut dj = vec![0.0f64; naux2];
        let djc = &self.dj1[comp * naux2..(comp + 1) * naux2];
        for p in 0..naux {
            if self.aux_atom[p] != atm {
                continue;
            }
            for q in 0..naux {
                dj[p + q * naux] -= djc[p + q * naux];
                dj[q + p * naux] -= djc[p + q * naux];
            }
        }
        dj
    }
}

// ---------------------------------------------------------------------------
// small dense helpers (flat col-major buffers, first index fastest)
// ---------------------------------------------------------------------------

