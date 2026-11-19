//! Pair-list ("materialised AO-pair kernel") BSE matvec.
//!
//! Selected with `bse_matvec_style = "pair"` in `[quasiparticle_methods]`,
//! together with `bse_pair_screening_tol`.
//!
//! # Why an AO-pair kernel
//!
//! The historical (`matvec`) and AO-fold (`matvec_fast`) routes both keep the
//! auxiliary index `q` open and contract it inside every matvec application:
//!
//! ```text
//! (A z)[i,a] = ... - f * sum_q (Xo^T T^q Xo) (Xv^T J^q Xv) z
//! ```
//!
//! so the per-application cost carries a factor `n_aux`.  Because the RI
//! three-centre tensor is a *metric-fitted* expansion, both the bare and the
//! screened AO kernels collapse onto the **AO pair** `(mu,nu)`:
//!
//! ```text
//! W[(mu,nu),(la,si)] = sum_q T^q[mu,nu] J^q[la,si]      (screened, "exchange")
//! ```
//!
//! and the exchange part of the BSE operator becomes an ordinary four-index
//! contraction in the AO basis:
//!
//! ```text
//! K[mu,la] = sum_{nu,si} W[(mu,nu),(la,si)] P[nu,si],   P = Xo Z Xv^T
//! W_A = Xo^T K   Xv          (A block)
//! W_B = Xo^T K^T Xv          (B block)
//! ```
//!
//! `W_B` reuses `K` transposed, which is why the two blocks share one kernel.
//! This is the AO-direct exchange of TDDFT/DFT, `K[mu,la] =
//! sum_{nu,si}(mu nu|la si) P[nu,si]`; the only difference is that an RI
//! density fitting supplies the two-electron integrals.
//!
//! # The approximation
//!
//! Only AO pairs with Schwarz factor
//!
//! ```text
//! Q[mu,nu] = || R[(mu,nu), :] ||_2 = sqrt(sum_q (mu nu|q)^2) ~ sqrt((mu nu|mu nu))
//! ```
//!
//! above `bse_pair_screening_tol` are kept.  `tol = 0` keeps every pair whose RI
//! row is not exactly zero, which reproduces the AO-fold route up to summation
//! order (a dropped row is exactly zero in `W`, so nothing is approximated).
//!
//! # Cost
//!
//! With `L` kept pairs and `M = n_aux`:
//!
//! ```text
//! build   : L^2 * M  (the pair kernel)  +  L * M^2 (the dielectric contraction)
//! storage : L^2 doubles
//! matvec  : 4 L^2  multiply-adds (kernel)  +  4 N O V (AO <-> MO round trips)
//!           + 2 M O V (direct term, which stays in the auxiliary basis)
//! ```
//!
//! The matvec no longer scales with `M`.  The price is the `L^2 M` build, whose
//! break-even against the AO-fold route (`M O V (O+V)` per application) sits at
//! roughly
//!
//! ```text
//! n_matvec ~ (L^2 M) / (M O V (O+V) - 4 L^2 - 4 N O V)
//! ```
//!
//! applications.  In practice the `L^2 M` term is two BLAS-3 GEMMs while the AO
//! build folds `M N^2 (O+V)` in scalar loops, so on the water chains measured in
//! `docs/bse_pair_kernel.md` the pair build is already the *cheaper* of the two
//! from `n = 6` onwards and the route dominates on both axes.
//!
//! # Kernel organisation (cache blocking)
//!
//! The kernel is a *matrix-vector* product in the compressed pair index with a
//! scatter into the `N x N` AO matrix, so its arithmetic intensity is bounded by
//! `4 L^2` multiply-adds per `8 L^2` bytes of `W`.  The implementation
//!
//! * streams `W` exactly once, one contiguous row per row-pair;
//! * keeps the column pair tables `(pa, pb)` out of the inner streaming loop by
//!   making them the *outer* loop, so they are read once per row block instead
//!   of once per `W` entry;
//! * blocks the row pairs so the `rows * L` slab of `W` fits in last-level cache
//!   and each cache line is consumed eight times (once per `f64` in it);
//! * keeps the AO transition density `P` (`N^2` doubles, row-major for
//!   unit-stride gathers) and the `N x N` accumulator hot while a row block runs;
//! * accumulates the two `+= W P[., y]` terms of a constant-`v` run in registers
//!   and stores once, which removes a store-to-load-forwarding chain on the
//!   accumulator element (worth 1.37x on the 4-water chain).
//!
//! The full derivation, cost model and single-core measurements are in
//! `docs/bse_pair_kernel.md`.

use rayon::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, omp_set_num_threads_wrapper};
use rest_tensors::matrix::matrixfullslice::{MatrixFullSlice, MatrixFullSliceMut};
use rest_tensors::matrix::MatrixFull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_bse::matvec_fast::build_rect_fold;
use crate::ri_gw::get_occ_params_per_spin;
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;

/// Default ceiling on `8 L^2` (the kernel matrix), in megabytes.  Overridable
/// with `REST_BSE_PAIR_MAX_MB`.
const DEFAULT_MAX_MB: f64 = 4096.0;

/// Row pairs per kernel block.  With [`BC`] this fixes the staged `W` tile at
/// `RI * BC` doubles (64 KiB for the values below), which is L2 resident.
const RI: usize = 64;
/// Column pairs per kernel block.
const BC: usize = 128;

struct PairSpin {
    n_o: usize,
    n_v: usize,
    n_state: usize,
    start_mo: usize,
    lumo: usize,
    /// `X[mu, A]`, `[n_bas, n_mo]` column-major.
    x: Vec<f64>,
    /// `R[q][i][b] = sum_{mu,nu} J^q[mu,nu] Xo[mu,i] Xv[nu,b]` — the **un**screened
    /// occupied x virtual fold used by the direct term, `r[q*V*O + b*O + i]`.
    r: Vec<f64>,
}

/// Cached `z`-dependent terms, so that the `a`-then-`b` pattern of
/// `BseMatvec::apb` / `BseMatvec::amb` pays for the pair kernel once.
struct Memo {
    z: Vec<f64>,
    wa: Vec<f64>,
    wb: Vec<f64>,
    v2: Vec<f64>,
}

/// The pair-list BSE matvec context.
pub struct PairBseContext {
    n_bas: usize,
    n_aux: usize,
    n_pairs: usize,
    /// Number of leading off-diagonal entries in the pair list (see `build`).
    n_off: usize,
    /// AO indices of the kept pairs, `pa[p] <= pb[p]`; all `pa[p] != pb[p]`
    /// entries come first.
    pa: Vec<u32>,
    pb: Vec<u32>,
    /// `W[p*L + c]`, row-major readable (see `build` for why the column-major
    /// BLAS result is already in this order).
    w: Vec<f64>,
    spins: Vec<PairSpin>,
    exchange_rescaling: f64,
    blas_threads: usize,
    /// `max|W - W^T| / max|W|` (diagnostic; the kernel never assumes symmetry).
    pub kernel_asymmetry: f64,
    /// Seconds spent in [`PairBseContext::build`].
    pub build_seconds: f64,
    /// Bytes of persistent storage.
    pub bytes: usize,
    matvec_count: AtomicUsize,
    memo: Mutex<Option<Memo>>,
}

impl PairBseContext {
    /// Build using `bse_pair_screening_tol` from the control file.
    pub fn build_from_ctrl(scf_data: &SCF, inverse_dielectric: &MatrixFull<f64>) -> Option<Self> {
        let qp = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap_or_default();
        Self::build(scf_data, inverse_dielectric, qp.bse_pair_screening_tol)
    }

    /// Build the context.
    ///
    /// Returns `None` -- with a printed reason -- when the packed RI tensor is
    /// missing, the reference is not restricted, the orbital window is
    /// inconsistent, or the kernel would exceed the memory ceiling.
    pub fn build(
        scf_data: &SCF,
        inverse_dielectric: &MatrixFull<f64>,
        screening_tol: f64,
    ) -> Option<Self> {
        let start = Instant::now();
        let n_bas = scf_data.mol.num_basis;
        let n_aux = inverse_dielectric.size[0];
        let n_spin = scf_data.mol.spin_channel.max(1);
        if n_spin != 1 {
            println!(
                "bse_matvec_style = \"pair\": the pair-list matvec is implemented for the \
                 restricted reference only; keeping the other backend."
            );
            return None;
        }
        let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap_or_default();

        let (rimatr, _, _) = match scf_data.rimatr_bse.as_ref().or(scf_data.rimatr.as_ref()) {
            Some(v) => v,
            None => {
                println!(
                    "bse_matvec_style = \"pair\" needs the packed AO-basis RI tensor \
                     (rimatr_bse, or the regular rimatr when no BSE auxiliary basis is set): \
                     set use_ri_symm = true. Keeping the other backend."
                );
                return None;
            }
        };
        let n_packed = n_bas * (n_bas + 1) / 2;
        if rimatr.size[0] != n_packed || rimatr.size[1] != n_aux {
            println!(
                "bse_matvec_style = \"pair\": rimatr has shape [{}, {}], expected [{}, {}]; \
                 keeping the other backend.",
                rimatr.size[0], rimatr.size[1], n_packed, n_aux
            );
            return None;
        }
        let rdata = &rimatr.data;

        // ---- Schwarz factors and the kept pair list --------------------------
        let mut qv = vec![0.0f64; n_packed];
        qv.par_chunks_mut(1024).enumerate().for_each(|(blk, out)| {
            let k0 = blk * 1024;
            for (i, o) in out.iter_mut().enumerate() {
                let mut s = 0.0f64;
                for q in 0..n_aux {
                    let v = rdata[k0 + i + q * n_packed];
                    s += v * v;
                }
                *o = s.sqrt();
            }
        });
        let mut packed_index: Vec<u32> = Vec::new();
        let mut pa: Vec<u32> = Vec::new();
        let mut pb: Vec<u32> = Vec::new();
        {
            let mut k = 0usize;
            for mu in 0..n_bas {
                for nu in 0..=mu {
                    if qv[k] > screening_tol {
                        packed_index.push(k as u32);
                        pa.push(nu as u32);
                        pb.push(mu as u32);
                    }
                    k += 1;
                }
            }
        }
        drop(qv);
        // Order the pair list with **all off-diagonal pairs first**.  The kernel
        // then runs its hot tile with `x != u` and `y != v` guaranteed, so the
        // inner loop is branch-free; the `|D| <= N` diagonal pairs are handled
        // by a short separate pass.
        let n_pairs_all = pa.len();
        let mut order: Vec<u32> = (0..n_pairs_all as u32).collect();
        order.sort_by_key(|&i| (pa[i as usize] == pb[i as usize]) as u8);
        let n_off = order
            .iter()
            .position(|&i| pa[i as usize] == pb[i as usize])
            .unwrap_or(n_pairs_all);
        let pa: Vec<u32> = order.iter().map(|&i| pa[i as usize]).collect();
        let pb: Vec<u32> = order.iter().map(|&i| pb[i as usize]).collect();
        let packed_index: Vec<u32> =
            order.iter().map(|&i| packed_index[i as usize]).collect();
        drop(order);
        let n_pairs = pa.len();
        if n_pairs == 0 {
            println!(
                "bse_matvec_style = \"pair\": screening removed every AO pair; keeping the \
                 other backend."
            );
            return None;
        }
        let max_mb = std::env::var("REST_BSE_PAIR_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(DEFAULT_MAX_MB);
        let kernel_mb = (n_pairs as f64) * (n_pairs as f64) * 8.0 / 1048576.0;
        if kernel_mb > max_mb {
            println!(
                "bse_matvec_style = \"pair\": the pair kernel would need {:.1} MB \
                 (L = {} pairs) which exceeds the {:.0} MB ceiling \
                 (raise REST_BSE_PAIR_MAX_MB to override); keeping the other backend.",
                kernel_mb, n_pairs, max_mb
            );
            return None;
        }
        println!(
            "BSE matvec pair list: kept {} / {} AO pairs ({:.2}%) at tol = {:.1e}; \
             kernel {:.1} MB",
            n_pairs,
            n_packed,
            100.0 * n_pairs as f64 / n_packed as f64,
            screening_tol,
            kernel_mb
        );

        // ---- R (kept rows of the packed RI tensor) and W = R D^T R^T ---------
        //
        // `Rm[i, q] = J^q[pair i]`, `Rtm = Rm D^T` so `Rtm[i,q] = T^q[pair i]`,
        // and `W[p, c] = sum_q Rtm[p,q] Rm[c,q]`.
        //
        // `_dgemm_full(A, 'N', B, 'T', C)` writes `C[i,j] = sum_q A[i,q]B[j,q]`
        // column-major, i.e. at `C.data[j*L + i]`.  With `A = Rm`, `B = Rtm`
        // that puts `sum_q Rm[c,q]Rtm[p,q] = W[p,c]` at `data[p*L + c]`, which is
        // exactly `W` **row-major**: no transposition is needed.
        let mut rm = MatrixFull::new([n_pairs, n_aux], 0.0);
        rm.data.par_chunks_mut(n_pairs).enumerate().for_each(|(q, col)| {
            let base = q * n_packed;
            for (i, o) in col.iter_mut().enumerate() {
                *o = rdata[packed_index[i] as usize + base];
            }
        });
        let mut rtm = MatrixFull::new([n_pairs, n_aux], 0.0);
        _dgemm_full(&rm, 'N', inverse_dielectric, 'T', &mut rtm, 1.0, 0.0);
        let mut w = vec![0.0f64; n_pairs * n_pairs];
        {
            let sw = [n_pairs, n_pairs];
            let iw = [1, n_pairs];
            let mut w_view = MatrixFullSliceMut { size: &sw, indicing: &iw, data: &mut w };
            _dgemm_full(&rm, 'N', &rtm, 'T', &mut w_view, 1.0, 0.0);
        }
        drop(rm);
        drop(rtm);

        // Diagnostic only: the kernel does not assume symmetry, but a symmetric
        // inverse dielectric (the normal case) makes `W` symmetric, which is
        // what would allow a packed-triangle kernel later.
        let wmax = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut asym = 0.0f64;
        for p in 0..n_pairs {
            for c in 0..p {
                let d = (w[p * n_pairs + c] - w[c * n_pairs + p]).abs();
                if d > asym {
                    asym = d;
                }
            }
        }
        let kernel_asymmetry = if wmax > 0.0 { asym / wmax } else { 0.0 };

        // ---- per-spin data ---------------------------------------------------
        let occ = get_occ_params_per_spin(scf_data, 'N');
        let (start_mo, n_state, n_o, n_v, homo, lumo) = if n_spin == 1 {
            get_occupation_parameters(scf_data, 'N')
        } else {
            let o = occ[0];
            (o.start_mo, o.num_state, o.occ_size, o.vir_size, o.homo, o.lumo)
        };
        let eig = &scf_data.eigenvectors[0];
        if eig.size[0] != n_bas
            || n_o == 0
            || n_v == 0
            || homo + 1 - start_mo != n_o
            || eig.size[1] < lumo + n_v
        {
            println!(
                "bse_matvec_style = \"pair\": inconsistent orbital window (n_bas={}, n_mo={}, \
                 start_mo={}, homo={}, lumo={}, n_o={}, n_v={}); keeping the other backend.",
                eig.size[0], eig.size[1], start_mo, homo, lumo, n_o, n_v
            );
            return None;
        }
        let x = eig.data.clone();
        let xo = &x[start_mo * n_bas..(homo + 1) * n_bas];
        let xv = &x[lumo * n_bas..(lumo + n_v) * n_bas];

        // The direct term stays in the auxiliary basis: `r` is only `M*O*V` and
        // costs `2 M O V` per application, far below the `L^2` pair kernel, so a
        // second (unscreened) `L x L` kernel is not worth materialising.
        let mut r = vec![0.0f64; n_aux * n_v * n_o];
        {
            let nb = ((4usize << 20) / (n_bas * n_bas * 8).max(1)).clamp(1, 64);
            let slab_bytes = nb * n_bas * n_bas;
            r.par_chunks_mut(nb * n_v * n_o)
                .enumerate()
                .for_each_init(
                    || (vec![0.0f64; slab_bytes], Default::default()),
                    |(slab, scratch), (blk, ob)| {
                        let q0 = blk * nb;
                        let this = (n_aux - q0).min(nb);
                        build_rect_fold(
                            ob, rdata, n_packed, n_bas, q0, this, xo, n_o, xv, n_v, &None,
                            &None, slab, scratch,
                        );
                    },
                );
        }

        let spins = vec![PairSpin { n_o, n_v, n_state, start_mo, lumo, x, r }];
        let bytes = w.len() * 8
            + spins.iter().map(|c| (c.x.len() + c.r.len()) * 8).sum::<usize>()
            + (pa.len() + pb.len()) * 4;
        let build_seconds = start.elapsed().as_secs_f64();
        println!(
            "BSE matvec (pair list): L = {} pairs, M = {}, kernel {:.1} MB, total {:.1} MB, \
             build {:.3} s, max|W - W^T|/max|W| = {:.2e}",
            n_pairs,
            n_aux,
            kernel_mb,
            bytes as f64 / 1048576.0,
            build_seconds,
            kernel_asymmetry
        );
        Some(Self {
            n_bas,
            n_aux,
            n_pairs,
            n_off,
            pa,
            pb,
            w,
            spins,
            exchange_rescaling: qp_ctrl.bse_exchange_rescaling,
            blas_threads: scf_data.mol.ctrl.num_threads.unwrap_or(1).max(1),
            kernel_asymmetry,
            build_seconds,
            bytes,
            matvec_count: AtomicUsize::new(0),
            memo: Mutex::new(None),
        })
    }

    // ------------------------------------------------------------- accessors --

    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }
    pub fn dims(&self) -> (usize, usize) {
        (self.spins[0].n_o, self.spins[0].n_v)
    }
    pub fn matvec_count(&self) -> usize {
        self.matvec_count.load(Ordering::Relaxed)
    }

    // ------------------------------------------------------- the pair kernel --

    /// Hot tile: off-diagonal row pairs against off-diagonal column pairs.
    ///
    /// # Safety
    ///
    /// `kax` and `kau` must point at two *distinct* rows of `ka`, each `n`
    /// doubles long; `px` and `pu` at two distinct rows of `pmat` (which must
    /// not overlap `ka`); every `yb[jj]` and `vb[jj]` must be `< n`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn tile_off_off(
        trow: &[f64],
        yb: &[u32],
        vb: &[u32],
        bc: usize,
        px: *const f64,
        pu: *const f64,
        kax: *mut f64,
        kau: *mut f64,
    ) {
        // The pair list is ordered by its *second* AO index, so `vb` is constant
        // over long runs.  Within a run the two `+= W P[., y]` terms all hit the
        // same accumulator element, and issuing them separately would leave a
        // store-to-load-forwarding chain on that element (measured: it dominates
        // the loop).  Accumulating them in registers and storing once per run
        // removes the chain; two partial sums per output keep the FMA latency
        // off the critical path as well.
        let mut cc = 0usize;
        while cc < bc {
            let v0 = *vb.get_unchecked(cc) as usize;
            let mut e = cc + 1;
            while e < bc && *vb.get_unchecked(e) as usize == v0 {
                e += 1;
            }
            let puv = *pu.add(v0);
            let pxv = *px.add(v0);
            let (mut s1a, mut s1b, mut s2a, mut s2b) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut c2 = cc;
            while c2 + 1 < e {
                let w0 = *trow.get_unchecked(c2);
                let w1 = *trow.get_unchecked(c2 + 1);
                let y0 = *yb.get_unchecked(c2) as usize;
                let y1 = *yb.get_unchecked(c2 + 1) as usize;
                *kax.add(y0) += w0 * puv;
                *kax.add(y1) += w1 * puv;
                *kau.add(y0) += w0 * pxv;
                *kau.add(y1) += w1 * pxv;
                s1a += w0 * *pu.add(y0);
                s2a += w0 * *px.add(y0);
                s1b += w1 * *pu.add(y1);
                s2b += w1 * *px.add(y1);
                c2 += 2;
            }
            while c2 < e {
                let w0 = *trow.get_unchecked(c2);
                let y0 = *yb.get_unchecked(c2) as usize;
                *kax.add(y0) += w0 * puv;
                *kau.add(y0) += w0 * pxv;
                s1a += w0 * *pu.add(y0);
                s2a += w0 * *px.add(y0);
                c2 += 1;
            }
            *kax.add(v0) += s1a + s1b;
            *kau.add(v0) += s2a + s2b;
            cc = e;
        }
    }

    /// One block of row pairs against every column pair.
    ///
    /// `ka` is row-major `[N, N]`; `pmat` is row-major `[N, N]` with
    /// `pmat[u*N + v] = P[u, v]`.
    ///
    /// For row pair `p = (x,u)` and column pair `c = (y,v)` the four
    /// contributions are
    ///
    /// ```text
    /// K[x,y] += W[p,c] P[u,v]      K[x,v] += W[p,c] P[u,y]
    /// K[u,y] += W[p,c] P[x,v]      K[u,v] += W[p,c] P[x,y]
    /// ```
    ///
    /// the first two from the ordered row pair `(x,u)`, the last two from
    /// `(u,x)`.  Diagonal row pairs (`x == u`) contribute only the first two,
    /// diagonal column pairs (`y == v`) only the first and third.
    ///
    /// # Blocking
    ///
    /// A `RI x BC` tile of `W` is staged into `tbuf` together with its `BC`
    /// column-pair indices, so the streaming loop reads `W` exactly once and
    /// touches the (large) `pa`/`pb` tables only `L / RI` times.  The AO rows of
    /// `P` and of the accumulator `K` are `N` doubles and stay in L1 for the
    /// whole tile, which is what keeps the four gathers per kernel entry cheap.
    /// Because the pair list puts every off-diagonal pair first, the hot tile
    /// needs no branching on `x == u` / `y == v` at all.
    #[allow(clippy::too_many_arguments)]
    fn kernel_rows(
        &self,
        i0: usize,
        i1: usize,
        pmat: &[f64],
        ka: &mut [f64],
        tbuf: &mut [f64],
        yb: &mut [u32],
        vb: &mut [u32],
    ) {
        let n = self.n_bas;
        let l = self.n_pairs;
        let n_off = self.n_off;
        let pa = &self.pa;
        let pb = &self.pb;
        let w = &self.w;

        // ---- hot path: off-diagonal rows x off-diagonal columns -------------
        let i1o = i1.min(n_off);
        if i0 < i1o {
            for c0 in (0..n_off).step_by(BC) {
                let bc = BC.min(n_off - c0);
                for jj in 0..bc {
                    yb[jj] = pa[c0 + jj];
                    vb[jj] = pb[c0 + jj];
                }
                for ii in i0..i1o {
                    let src = &w[ii * l + c0..ii * l + c0 + bc];
                    tbuf[(ii - i0) * BC..(ii - i0) * BC + bc].copy_from_slice(src);
                }
                for ii in i0..i1o {
                    let x = pa[ii] as usize;
                    let u = pb[ii] as usize;
                    let trow = &tbuf[(ii - i0) * BC..(ii - i0) * BC + bc];
                    // Two distinct accumulator rows and two distinct P rows.
                    let (kax, kau) = if x < u {
                        let (lo, hi) = ka.split_at_mut(u * n);
                        (&mut lo[x * n..x * n + n], &mut hi[..n])
                    } else {
                        let (lo, hi) = ka.split_at_mut(x * n);
                        (&mut hi[..n], &mut lo[u * n..u * n + n])
                    };
                    // SAFETY: `x != u` (off-diagonal pair) so the two rows of
                    // `ka` are disjoint; `px`/`pu` are rows of `pmat`, a
                    // different buffer; `pa`/`pb` entries are `< n` by
                    // construction.
                    unsafe {
                        Self::tile_off_off(
                            trow,
                            yb,
                            vb,
                            bc,
                            pmat[x * n..x * n + n].as_ptr(),
                            pmat[u * n..u * n + n].as_ptr(),
                            kax.as_mut_ptr(),
                            kau.as_mut_ptr(),
                        );
                    }
                }
            }
        }

        // ---- diagonal row pairs against the off-diagonal column pairs -------
        //
        // A diagonal row pair `(x,x)` has a single ordered form, so it
        // contributes the two terms
        //
        // ```text
        // K[x,y] += W[p,c] P[x,v]      K[x,v] += W[p,c] P[x,y]
        // ```
        //
        // both of which land in the *same* accumulator row.
        if i1 > n_off {
            for ii in i0.max(n_off)..i1 {
                let x = pa[ii] as usize;
                let px = &pmat[x * n..x * n + n];
                let kax = &mut ka[x * n..x * n + n];
                for (c, wv) in w[ii * l..ii * l + n_off].iter().enumerate() {
                    if *wv == 0.0 {
                        continue;
                    }
                    let y = pa[c] as usize;
                    let v = pb[c] as usize;
                    kax[y] += *wv * px[v];
                    kax[v] += *wv * px[y];
                }
            }
        }

        // ---- tail: the `|D| <= N` diagonal column pairs ---------------------
        for ii in i0..i1 {
            let x = pa[ii] as usize;
            let u = pb[ii] as usize;
            let wrow = &w[ii * l + n_off..ii * l + l];
            let px = &pmat[x * n..x * n + n];
            if x == u {
                let kax = &mut ka[x * n..x * n + n];
                for (b, wv) in wrow.iter().enumerate() {
                    if *wv == 0.0 {
                        continue;
                    }
                    let y = pa[n_off + b] as usize;
                    kax[y] += *wv * px[y];
                }
            } else {
                let pu = &pmat[u * n..u * n + n];
                let (kax, kau) = if x < u {
                    let (lo, hi) = ka.split_at_mut(u * n);
                    (&mut lo[x * n..x * n + n], &mut hi[..n])
                } else {
                    let (lo, hi) = ka.split_at_mut(x * n);
                    (&mut hi[..n], &mut lo[u * n..u * n + n])
                };
                for (b, wv) in wrow.iter().enumerate() {
                    if *wv == 0.0 {
                        continue;
                    }
                    let y = pa[n_off + b] as usize;
                    kax[y] += *wv * pu[y];
                    kau[y] += *wv * px[y];
                }
            }
        }
    }

    /// `K[mu,la] = sum_{nu,si} W[(mu,nu),(la,si)] P[nu,si]`.
    ///
    /// Returns `(K, K^T)`, both column-major `[N,N]` (the form BLAS wants).  The
    /// AO kernel is symmetric under exchanging its two pair indices, so the B
    /// block is just the same contraction against the transpose.
    pub fn apply_kernel(&self, pmat: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_bas;
        let l = self.n_pairs;
        let n_threads = rayon::current_num_threads().max(1);
        // Parallelising over row blocks needs a partial `K` per worker (row
        // blocks write to overlapping AO rows); only do it when that fits.
        let partial_mb = (n_threads * n * n * 8) as f64 / 1048576.0;
        let parallel = n_threads > 1 && partial_mb <= 512.0;
        let mut ka = if parallel {
            (0..l)
                .into_par_iter()
                .step_by(RI)
                .fold(
                    || {
                        omp_set_num_threads_wrapper(1);
                        (
                            vec![0.0f64; n * n],
                            vec![0.0f64; RI * BC],
                            vec![0u32; BC],
                            vec![0u32; BC],
                        )
                    },
                    |(mut acc, mut tb, mut yb, mut vb), i0| {
                        self.kernel_rows(
                            i0,
                            (i0 + RI).min(l),
                            pmat,
                            &mut acc,
                            &mut tb,
                            &mut yb,
                            &mut vb,
                        );
                        (acc, tb, yb, vb)
                    },
                )
                .map(|(acc, _, _, _)| acc)
                .reduce(
                    || vec![0.0f64; n * n],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b.iter()) {
                            *x += y;
                        }
                        a
                    },
                )
        } else {
            let mut acc = vec![0.0f64; n * n];
            let mut tb = vec![0.0f64; RI * BC];
            let mut yb = vec![0u32; BC];
            let mut vb = vec![0u32; BC];
            let mut i0 = 0;
            while i0 < l {
                self.kernel_rows(i0, (i0 + RI).min(l), pmat, &mut acc, &mut tb, &mut yb, &mut vb);
                i0 += RI;
            }
            acc
        };
        omp_set_num_threads_wrapper(self.blas_threads);
        // `ka` is row-major `K`; a column-major `[N,N]` buffer is its transpose,
        // so one copy yields both `K` (column-major) and `K^T` (column-major).
        let mut k_cm = vec![0.0f64; n * n];
        for x in 0..n {
            let row = &ka[x * n..x * n + n];
            for (y, val) in row.iter().enumerate() {
                k_cm[y * n + x] = *val;
            }
        }
        // After the swap: `ka` = column-major `K`, `k_cm` = the old row-major
        // buffer, which read as a column-major matrix is `K^T`.
        std::mem::swap(&mut ka, &mut k_cm);
        (ka, k_cm)
    }

    /// `P[nu,si] = sum_{i,a} Xo[nu,i] z[i + a*O] Xv[si,a]`, returned row-major
    /// (`p[u*N + v] = P[u,v]`).
    fn transition_density(&self, c: &PairSpin, z: &[f64]) -> Vec<f64> {
        let n = self.n_bas;
        let (o, v) = (c.n_o, c.n_v);
        let szo = [o, v];
        let izo = [1, o];
        let z_view = MatrixFullSlice { size: &szo, indicing: &izo, data: z };
        let sxo = [n, o];
        let ixo = [1, n];
        let xo_view =
            MatrixFullSlice { size: &sxo, indicing: &ixo, data: &c.x[c.start_mo * n..(c.start_mo + o) * n] };
        let sxv = [n, v];
        let ixv = [1, n];
        let xv_view =
            MatrixFullSlice { size: &sxv, indicing: &ixv, data: &c.x[c.lumo * n..(c.lumo + v) * n] };
        // y[nu, a] = sum_i Xo[nu,i] z[i,a]
        let mut y = MatrixFull::new([n, v], 0.0);
        _dgemm_full(&xo_view, 'N', &z_view, 'N', &mut y, 1.0, 0.0);
        // pt[i,j] = sum_a Xv[i,a] y[j,a] = P[j,i]; column-major storage puts
        // `P[u,v]` at `data[u*N + v]`, i.e. `pt.data` is row-major `P`.
        let mut pt = MatrixFull::new([n, n], 0.0);
        _dgemm_full(&xv_view, 'N', &y, 'T', &mut pt, 1.0, 0.0);
        pt.data
    }

    /// `(W_A, W_B, V2)` for one transition vector, memoised so that the
    /// `a`-then-`b` call pattern pays for the pair kernel once.
    fn terms(&self, z: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut guard = self.memo.lock().unwrap();
        if let Some(m) = guard.as_ref() {
            if m.z == z {
                return (m.wa.clone(), m.wb.clone(), m.v2.clone());
            }
        }
        let c = &self.spins[0];
        let n = self.n_bas;
        let (o, v) = (c.n_o, c.n_v);
        let p = self.transition_density(c, z);
        let (k_cm, kt_cm) = self.apply_kernel(&p);
        drop(p);
        let sxo = [n, o];
        let ixo = [1, n];
        let xo_view =
            MatrixFullSlice { size: &sxo, indicing: &ixo, data: &c.x[c.start_mo * n..(c.start_mo + o) * n] };
        let sxv = [n, v];
        let ixv = [1, n];
        let xv_view =
            MatrixFullSlice { size: &sxv, indicing: &ixv, data: &c.x[c.lumo * n..(c.lumo + v) * n] };
        let sk = [n, n];
        let ik = [1, n];
        let mut wa = MatrixFull::new([o, v], 0.0);
        let mut wb = MatrixFull::new([o, v], 0.0);
        let mut tmp = MatrixFull::new([o, n], 0.0);
        // W_A = Xo^T K   Xv
        let kv = MatrixFullSlice { size: &sk, indicing: &ik, data: &k_cm };
        _dgemm_full(&xo_view, 'T', &kv, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', &xv_view, 'N', &mut wa, 1.0, 0.0);
        // W_B = Xo^T K^T Xv
        let ktv = MatrixFullSlice { size: &sk, indicing: &ik, data: &kt_cm };
        _dgemm_full(&xo_view, 'T', &ktv, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', &xv_view, 'N', &mut wb, 1.0, 0.0);
        drop(k_cm);
        drop(kt_cm);
        let v2 = self.direct_term(z);
        let out = (wa.data, wb.data, v2);
        *guard = Some(Memo { z: z.to_vec(), wa: out.0.clone(), wb: out.1.clone(), v2: out.2.clone() });
        out
    }

    /// `V2[i,a] = sum_q s_q R_q[i,a]`, `s_q = sum_{j,b} R_q[j,b] z[j + b*O]`.
    fn direct_term(&self, z: &[f64]) -> Vec<f64> {
        let c = &self.spins[0];
        let (o, v, m) = (c.n_o, c.n_v, self.n_aux);
        let nov = o * v;
        let mut result = vec![0.0f64; nov];
        for q in 0..m {
            let rq = &c.r[q * v * o..(q + 1) * v * o];
            let mut s = 0.0f64;
            for a in 0..v {
                let ra = &rq[a * o..(a + 1) * o];
                let za = &z[a * o..(a + 1) * o];
                for j in 0..o {
                    s += ra[j] * za[j];
                }
            }
            if s == 0.0 {
                continue;
            }
            for a in 0..v {
                let ra = &rq[a * o..(a + 1) * o];
                let res = &mut result[a * o..(a + 1) * o];
                for i in 0..o {
                    res[i] += s * ra[i];
                }
            }
        }
        result
    }

    /// The direct-term coefficient, matching `matvec_fast::spin_letter` exactly:
    /// `'S' -> 2`, `'T' -> 0`, **everything else** (including `"rpa"` and the
    /// default `"none"`) `-> 1`.  Anything narrower here silently drops the
    /// direct term for unrecognised spellings.
    fn direct_coefficient(qp_ctrl: &QuasiParticle) -> f64 {
        match qp_ctrl.bse_spin.as_str() {
            "singlet" => 2.0,
            "triplet" => 0.0,
            _ => 1.0,
        }
    }

    // ------------------------------------------------------------ test hooks --

    /// `P = Xo Z Xv^T` for the (single) spin channel, row-major `[N,N]`.
    pub fn transition_density_of(&self, z: &[f64]) -> Vec<f64> {
        self.transition_density(&self.spins[0], z)
    }

    /// `(pa[p], pb[p])` of kept pair `p`.
    pub fn pair_of(&self, p: usize) -> (usize, usize) {
        (self.pa[p] as usize, self.pb[p] as usize)
    }

    /// `W[p, c]` of the materialised kernel.
    pub fn w_at(&self, p: usize, c: usize) -> f64 {
        self.w[p * self.n_pairs + c]
    }

    pub fn n_aux(&self) -> usize {
        self.n_aux
    }

    /// A-block matvec, restricted reference.
    pub fn a_block_matvec(&self, scf_data: &SCF, qp_ctrl: &QuasiParticle, z: &[f64]) -> Vec<f64> {
        self.matvec_count.fetch_add(1, Ordering::Relaxed);
        let c = &self.spins[0];
        let (o, v) = (c.n_o, c.n_v);
        let nov = o * v;
        assert_eq!(z.len(), nov);
        let qp = &scf_data.gwqp.0;
        let mut result = vec![0.0f64; nov];
        for a in 0..v {
            let ea = qp[c.lumo + a];
            for i in 0..o {
                result[i + a * o] = (ea - qp[c.start_mo + i]) * z[i + a * o];
            }
        }
        let (wa, _, v2) = self.terms(z);
        let f = self.exchange_rescaling;
        for k in 0..nov {
            result[k] -= f * wa[k];
        }
        let cc = Self::direct_coefficient(qp_ctrl);
        if cc != 0.0 {
            for k in 0..nov {
                result[k] += cc * v2[k];
            }
        }
        result
    }

    /// B-block matvec.  Reproduces the historical asymmetries: no
    /// `bse_exchange_rescaling`, and no direct term for triplets.
    pub fn b_block_matvec(&self, _scf_data: &SCF, qp_ctrl: &QuasiParticle, z: &[f64]) -> Vec<f64> {
        self.matvec_count.fetch_add(1, Ordering::Relaxed);
        let c = &self.spins[0];
        let (o, v) = (c.n_o, c.n_v);
        let nov = o * v;
        assert_eq!(z.len(), nov);
        let (_, wb, v2) = self.terms(z);
        let mut result = vec![0.0f64; nov];
        for k in 0..nov {
            result[k] = -wb[k];
        }
        let cc = Self::direct_coefficient(qp_ctrl);
        if cc != 0.0 {
            for k in 0..nov {
                result[k] += cc * v2[k];
            }
        }
        result
    }
}
