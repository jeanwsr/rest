fn release_vec<T>(cell: &std::cell::RefCell<Vec<T>>) {
    if let Ok(mut g) = cell.try_borrow_mut() {
        g.clear();
        g.shrink_to_fit();
    }
}

const FOLD_BLK: usize = 128;

/// Fixed work-block size (**transition index** `ia`) of the parallel polar
/// pullback.
///
/// The blocks are laid out from the index alone and the partials are merged in
/// block order, so the result is independent of the thread count.
const POLAR_BLK: usize = 128;

/// Free-function core of [`GwGradEngine::polar_pullback_parallel`], so the
/// parallel closure captures plain slices instead of the engine (`&GwGradEngine`
/// is not `Sync` because of its `RefCell` caches).
#[allow(clippy::too_many_arguments)]
fn polar_pullback_blocks(
    naux: usize,
    nmo: usize,
    nocc: usize,
    qia: &[f64],
    de_ia: &[f64],
    bq: &mut [Vec<f64>],
    be: &mut [Vec<f64>],
    k: usize,
    v: &[f64],
    d: &[f64],
    dd: &[f64],
    packed: bool,
) {
    use rayon::prelude::*;
    let nov = de_ia.len();
    let vm = to_mat(v, naux, naux);
    let qiam = to_mat(qia, naux, nov);
    let spans = polar_blocks(nov);
    let bqk = &mut bq[k];
    let bek = &mut be[k];
    let parts: Vec<(usize, Vec<f64>, Vec<f64>)> = spans
        .par_iter()
        .map(|&[c0, c1]| {
            let nc = c1 - c0;
            let qm = to_mat(&qiam.data[c0 * naux..c1 * naux], naux, nc);
            let mut vq = MatrixFull::new([naux, nc], 0.0);
            _dgemm_full(&vm, 'N', &qm, 'N', &mut vq, 1.0, 0.0);
            let mut dbe = vec![0.0f64; nmo];
            let mut dbq = vec![0.0f64; nc * naux];
            for (lc, col) in (c0..c1).enumerate() {
                let i = col % nocc;
                let a = nocc + col / nocc;
                let mut bg = 0.0f64;
                for p in 0..naux {
                    let vqv = vq.data[p + lc * naux];
                    dbq[lc * naux + p] = 2.0 * d[col] * vqv;
                    bg += qiam.data[p + col * naux] * vqv * dd[col];
                }
                dbe[i] -= bg;
                dbe[a] += bg;
            }
            (c0, dbq, dbe)
        })
        .collect();
    // ordered merge: the accumulation order is the block index alone
    let mut be_local = vec![0.0f64; nmo];
    for (c0, dbq, dbe) in parts {
        let nc = dbq.len() / naux;
        for (lc, col) in (c0..c0 + nc).enumerate() {
            let i = col % nocc;
            let a = nocc + col / nocc;
            let col_pair = if packed {
                packed_pair_index(nmo, i, a)
            } else {
                i + a * nmo
            } * naux;
            for p in 0..naux {
                bqk[col_pair + p] += dbq[lc * naux + p];
            }
        }
        for (row, val) in dbe.iter().enumerate() {
            be_local[row] += *val;
        }
    }
    for (row, val) in be_local.iter().enumerate() {
        bek[row] += *val;
    }
}

/// The fixed `[ia0, ia1)` blocks of the parallel polar pullback.
#[inline]
fn polar_blocks(n: usize) -> Vec<[usize; 2]> {
    index_blocks(n, POLAR_BLK)
}

/// Fixed `[c0, c1)` index blocks of width `blk`.
///
/// Used by the two column-separable pullback kernels (`response`, `gram`): their
/// output can be cut along an index that appears in no contraction, so the blocks
/// produce **disjoint** pieces of the result, need no reduction, and reproduce the
/// serial product bit for bit.  The layout depends on the index and the width
/// alone, never on the thread count.
#[inline]
fn index_blocks(n: usize, blk: usize) -> Vec<[usize; 2]> {
    (0..(n + blk - 1) / blk)
        .map(|b| [b * blk, ((b + 1) * blk).min(n)])
        .collect()
}

/// Column block of the metric response `S * corr` (the `nmo` index is not contracted).
const RESP_BLK: usize = 64;

/// Column block of the two Gram products (the `naux` output index is not contracted).
const GRAM_BLK: usize = 64;

/// Column block of the residue rank-1 updates (their `naux` output index is not
/// contracted either, so the split needs no reduction and no partial buffer).
const RES_BLK: usize = 128;

/// Upper bound on the number of *concurrent* workers of the memory-heavy
/// integral-driven phases (the streamed fold and the perturbation contraction).
///
/// Those phases evaluate one auxiliary shell (fold) or one auxiliary batch
/// (contraction) of raw integrals per task, and each task's block stays alive for
/// as long as the task runs, so the resident set grows by `workers * block`.  The
/// blocks are tens of MB at def2-TZVP, so an unbounded fan-out makes an
/// eight-thread run peak well above a one-thread run on the same job.  Working in
/// fixed chunks of `PAR_CAP` tasks keeps the extra resident set bounded and
/// independent of the machine's core count without changing any value: the partial
/// sums are still merged in index order.
const PAR_CAP: usize = 4;

/// [`PAR_CAP`], overridable through `REST_PAR_CAP`.
///
/// On a machine with a hard per-job memory limit the fan-out is the knob that
/// decides whether a large system fits at all (tetracene/def2-TZVP at 16
/// threads is 49.8 GB at `PAR_CAP = 4` and above the 134 GB job limit with no
/// cap at all), so the study that produced those numbers runs the largest
/// systems through this override instead of a rebuild.
fn par_cap() -> usize {
    std::env::var("REST_PAR_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(PAR_CAP)
}

// sub-step accumulators of the canonical-response assembly (seconds):
// [0] j_deriv_from, [1] k_deriv_from, [2] compute_j_upper, [3] compute_k_upper
thread_local! {
    static T_TIMER: std::cell::RefCell<[f64; 4]> = const { std::cell::RefCell::new([0.0; 4]) };
    /// `[response, gram, accumulate, bq/be]` of the streaming QP pullback
    /// (`REST_GRAD_TIMING=1`).
    static T_PULL: std::cell::RefCell<[f64; 9]> = const { std::cell::RefCell::new([0.0; 9]) };
}

fn timer_add(idx: usize, t0: Instant) {
    if grad_timing() {
        T_TIMER.with(|c| c.borrow_mut()[idx] += t0.elapsed().as_secs_f64());
    }
}

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

