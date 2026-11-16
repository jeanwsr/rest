// ---------------------------------------------------------------------------

/// One Cartesian component of the per-atom XC skeleton
/// (`hessian::rks::compute_vxc_h1ao`): the flattened col-major `[3*nao, nao]`
/// block reduced to a `[nao, nao]` col-major buffer of component `comp`
/// (rows `[comp*nao, (comp+1)*nao)`).
fn vxc_block_ao(flat: &[f64], comp: usize, nao: usize) -> Vec<f64> {
    let n2 = nao * nao;
    let mut out = vec![0.0f64; n2];
    let row0 = comp * nao;
    for j in 0..nao {
        for i in 0..nao {
            out[i + j * nao] = flat[row0 + i + j * 3 * nao];
        }
    }
    out
}

/// Temporary bisection mask for the DFT adjoint terms (debug only):
/// bit0 = XC kernel in `vtil`, bit1 = orbital-rotation part of `b_x` folded into
/// `Lambda`/`Lambda_s`, bit2 = kernel-response part of `b_x` (`-f_xc(D_bb)`) in
/// `L`, bit3 = the explicit `v_xc` skeleton in the `Lambda_ao` contraction.
/// Default 15 (all terms on).
fn dft_debug_mask() -> u32 {
    std::env::var("REST_ADJOINT_DFT_MASK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15)
}

fn nearest_grid_value(zs: &[f64], z: f64) -> f64 {
    let n = zs.len();
    if n == 0 {
        return z;
    }
    if z <= zs[0] {
        return zs[0];
    }
    if z >= zs[n - 1] {
        return zs[n - 1];
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
        zs[lo]
    } else {
        zs[hi]
    }
}

// ---------------------------------------------------------------------------
// automatic real-axis grid (best practice)
// ---------------------------------------------------------------------------

/// Range margin of the automatic rule over the largest residue frequency the
/// requested state window actually reads.
pub const AUTO_GRID_RANGE_MARGIN: f64 = 2.5;
/// Smallest automatic grid range (Ha).
pub const AUTO_GRID_RANGE_FLOOR: f64 = 0.5;
/// Target node spacing (Ha) of the automatic rule.  Measured on
/// H2O/def2-SVP/PBE with the range capped at `2.5 z_need`: the frontier grid
/// error is 0.45 / 0.67 / 2.2 / 0.009 / 0.008 / 0.001 eV at 6 / 12 / 24 / 48 /
/// 96 / 192 nodes, so ~0.025 Ha puts the uniform rule in its converged regime
/// (`h = 0.025` -> 51 nodes for that case).  Note the non-monotonicity at
/// n = 24: the residual is still sensitive to where a node falls relative to
/// the RPA poles of `W`, which no spacing rule removes.
pub const AUTO_GRID_DELTA: f64 = 0.025;
/// Smallest automatic node count (the historical default).
pub const AUTO_GRID_N_MIN: usize = 6;
/// Largest automatic node count (cost cap).
pub const AUTO_GRID_N_MAX: usize = 400;

/// Symmetry-degenerate tie set of orbital `idx` within `lo..hi`.
fn tie_set(e: &[f64], idx: usize, lo: usize, hi: usize, tol: f64) -> Vec<usize> {
    (lo..hi).filter(|&j| (e[j] - e[idx]).abs() <= tol).collect()
}

/// Largest residue frequency the state window `nsemin..=nsemax` reads, and the
/// transition that attains it.
///
/// A QP state `n` (with `omega ~ e_n`) has active poles `m` on its own branch
/// with `de = |e_m - e_n| > 0`, and the residue evaluates the screened response
/// at `z = |e_m - omega|`.  This is the frequency range the low-rank factors
/// have to resolve — the **core-inflated `de_max` of the scan is usually far
/// larger** (the scan pairs every pole with every scan centre), and spreading
/// the nodes over that unused range makes the grid error stagnate at
/// O(0.1-1 eV) instead of converging.
///
/// Returns `(z_need, (p, q, sign, p_on_occupied_branch))` with
/// `d z_need/dR = sign * (eps1[p] - eps1[q])`.
pub fn residue_z_need(
    e: &[f64],
    nocc: usize,
    nsemin: usize,
    nsemax: usize,
) -> (f64, Option<(usize, usize, f64, bool)>) {
    let nmo = e.len();
    let mut z_need = 0.0f64;
    let mut winner: Option<(usize, usize, f64, bool)> = None;
    for n in nsemin..=nsemax {
        let occ = n < nocc;
        let (lo, hi) = if occ { (0, nocc) } else { (nocc, nmo) };
        for m in lo..hi {
            let de = if occ { e[m] - e[n] } else { e[n] - e[m] };
            if de > 0.0 && de > z_need {
                z_need = de;
                // occupied: `de = e_m - e_n` grows with `e_m`
                // virtual  : `de = e_n - e_m` grows with `e_n`
                winner = if occ {
                    Some((m, n, 1.0, true))
                } else {
                    Some((n, m, 1.0, false))
                };
            }
        }
    }
    (z_need, winner)
}

/// Node count of the automatic rule for a grid range.
fn auto_n(range: f64) -> usize {
    (((range / AUTO_GRID_DELTA).ceil() as usize) + 1).clamp(AUTO_GRID_N_MIN, AUTO_GRID_N_MAX)
}

/// Best-practice automatic real-axis grid, used when
/// `GwGradConfig::nomega_chi_real == 0` / `QuasiParticle::nomega_chi_real == 0`.
///
/// The rule is
///
/// ```text
/// z_need = max_n max_{active m} |e_m - e_n|          (state window)
/// range  = clamp(AUTO_GRID_RANGE_MARGIN * z_need, AUTO_GRID_RANGE_FLOOR, de_max)
/// n      = clamp(1 + ceil(range / AUTO_GRID_DELTA), AUTO_GRID_N_MIN, AUTO_GRID_N_MAX)
/// ```
///
/// i.e. the node count grows **linearly with the frequency range the residues
/// read**, not with `de_max` and not linearly with the number of levels (which
/// is what dilates `de_max` non-linearly through the deep core).  A user-pinned
/// `omega_chi_max > 0` wins over the range part of the rule (`n` is still
/// derived from it).
///
/// Returns `(n_points, grid_scale, mover)`; `None` when the rule does not apply
/// (`omega_chi_max` pins a range *and* an explicit `n` was given).
#[allow(clippy::too_many_arguments)]
pub fn auto_real_axis_grid(
    e: &[f64],
    nocc: usize,
    nsemin: usize,
    nsemax: usize,
    de_max: f64,
    de_mover: Option<GridMover>,
    omega_chi_max: f64,
    nomega_chi_real: usize,
    tie_tol: f64,
) -> Option<(usize, f64, Option<GridMover>)> {
    if nomega_chi_real != 0 {
        return None;
    }
    if omega_chi_max > 0.0 {
        return Some((auto_n(omega_chi_max), omega_chi_max, None));
    }
    let (z_need, winner) = residue_z_need(e, nocc, nsemin, nsemax);
    let want = (AUTO_GRID_RANGE_MARGIN * z_need).max(AUTO_GRID_RANGE_FLOOR);
    if !(want < de_max) || winner.is_none() {
        // the scan range is already tighter than the rule: keep it (and its
        // mover, whose `scale` is the `de_max` one)
        return Some((auto_n(de_max), de_max, de_mover));
    }
    let (p, q, sign, p_occ) = winner.unwrap();
    let p_set = if p_occ {
        tie_set(e, p, 0, nocc, tie_tol)
    } else {
        tie_set(e, p, nocc, e.len(), tie_tol)
    };
    let m_set = tie_set(e, q, 0, e.len(), tie_tol);
    Some((
        auto_n(want),
        want,
        Some(GridMover {
            p_set,
            m_set,
            sign,
            scale: AUTO_GRID_RANGE_MARGIN,
        }),
    ))
}

/// REST real-axis grid construction: de_max scan + linear/quadratic nodes.
///
/// Besides the nodes themselves this returns their **geometry dependence**, so
/// that the gradient can differentiate the grid-mediated part of the residue
/// term exactly (see [`RealAxisGrid`]).
pub struct RealAxisGrid {
    /// the grid nodes, in increasing order
    pub nodes: Vec<f64>,
    /// `d node_i / d de_max` of every node (all zero when the grid is pinned by
    /// `omega_chi_max > 0`)
    pub node_sens: Vec<f64>,
    /// The transition energy that attains the scan maximum (see
    /// [`GridMover`]).  `None` when `de_max` does not move with the geometry
    /// (`omega_chi_max > 0`) or when the scan found no transition above
    /// `res_tol`.
    pub mover: Option<GridMover>,
}

/// The scan maximum `de_max = 1.05 * |e_p - (e_m + i*step_sigma)| + 0.1` of
/// [`build_real_axis_grid`], together with the symmetry-degenerate orbitals
/// that tie for it.
///
/// `de_max` is a **max** over the scan, so at a degenerate geometry it is not
/// differentiable: displacing by `+h` follows the tied branch with the largest
/// derivative and `-h` the one with the smallest, and the central difference of
/// a max is the average of those two branches.  Storing the tied sets lets the
/// gradient reproduce exactly that.
#[derive(Clone, Debug)]
pub struct GridMover {
    /// tied orbitals on the pole side (`de` grows with their energy)
    pub p_set: Vec<usize>,
    /// tied scan centres (`de` shrinks with their energy)
    pub m_set: Vec<usize>,
    /// `+1` when the maximum is an occupied pole (`de = e_p - omega`), `-1` when
    /// it is a virtual pole (`de = omega - e_a`)
    pub sign: f64,
    /// `d(grid_scale)/d(Delta e)` where `Delta e = e_p - e_m` for the pole pair
    /// above: `1.05` for the `de_max = 1.05*D + 0.1` scan, and the range margin
    /// when the automatic rule caps the range at a multiple of `z_need`.
    pub scale: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn build_real_axis_grid(
    e: &[f64],
    nocc: usize,
    nomega_chi_real: usize,
    nsemin: usize,
    nsemax: usize,
    nomega_sigma: usize,
    step_sigma: f64,
    res_tol: f64,
    grid_type: usize,
    omega_chi_max: f64,
    tie_tol: f64,
) -> RealAxisGrid {
    let nmo = e.len();
    let mut de_max = 0.0f64;
    let mut winner: Option<(usize, usize, f64)> = None;
    for mstate in nsemin..=nsemax {
        let e0 = e[mstate];
        for io in -(nomega_sigma as isize)..=(nomega_sigma as isize) {
            let omega = e0 + io as f64 * step_sigma;
            for p in 0..nocc {
                let de = e[p] - omega;
                if de > res_tol && de > de_max {
                    de_max = de;
                    winner = Some((p, mstate, 1.0));
                }
            }
            for a in nocc..nmo {
                let de = omega - e[a];
                if de > res_tol && de > de_max {
                    de_max = de;
                    winner = Some((a, mstate, -1.0));
                }
            }
        }
    }
    // Tie sets of the winner: the maximum is attained by a whole
    // symmetry-degenerate shell whenever one is involved.
    let mover = winner.map(|(p, m, sign)| {
        let mut p_set = Vec::new();
        let mut m_set = Vec::new();
        if sign > 0.0 {
            for i in 0..nocc {
                if (e[i] - e[p]).abs() <= tie_tol {
                    p_set.push(i);
                }
            }
        } else {
            for a in nocc..nmo {
                if (e[a] - e[p]).abs() <= tie_tol {
                    p_set.push(a);
                }
            }
        }
        for j in 0..nmo {
            if (e[j] - e[m]).abs() <= tie_tol {
                m_set.push(j);
            }
        }
        GridMover { p_set, m_set, sign, scale: 1.05 }
    });
    de_max = de_max * 1.05 + 0.1;
    // Automatic mode (`nomega_chi_real == 0`) or the historical layout.
    // `omega_chi_max > 0` pins the node frequencies: the grid then does not
    // move with the reference geometry at all.
    let (n_points, grid_scale, pinned, mover) = match auto_real_axis_grid(
        e,
        nocc,
        nsemin,
        nsemax,
        de_max,
        mover.clone(),
        omega_chi_max,
        nomega_chi_real,
        tie_tol,
    ) {
        Some((n, scale, mv)) => (n, scale, false, mv),
        None if omega_chi_max > 0.0 => (nomega_chi_real, omega_chi_max, true, None),
        None => (nomega_chi_real, de_max, false, mover),
    };
    let mut nodes = Vec::with_capacity(n_points);
    let mut node_sens = Vec::with_capacity(n_points);
    for i in 0..n_points {
        if n_points > 1 {
            let t = i as f64 / (n_points - 1) as f64;
            if grid_type == 1 {
                nodes.push(grid_scale * t * t);
                node_sens.push(if pinned { 0.0 } else { t * t });
            } else {
                nodes.push(grid_scale * t);
                node_sens.push(if pinned { 0.0 } else { t });
            }
        } else {
            nodes.push(grid_scale);
            node_sens.push(if pinned { 0.0 } else { 1.0 });
        }
    }
    RealAxisGrid {
        nodes,
        node_sens,
        mover,
    }
}

/// Compact a **pair-symmetric** `[nmo*nmo, naux]` buffer (pair-slowest, `pair = row + col*nmo`)
/// in place to its lower triangle `[nmo*(nmo+1)/2, naux]`, returning the packed length.
///
/// Written in increasing packed order, so the destination offset never overtakes an unread source
/// (`tp <= pair`), and the tail is released with `truncate` + `shrink_to_fit`. Used by the adjoint
/// gradient for `bq`, which the pullback has already symmetrised: the fold reads only the symmetric
/// plane, so rebuilding it from the packed form is exact while `naux*nmo^2/2` doubles (437 MB at
/// naphthalene/def2-TZVP) leave the resident set before the fold — the phase this gradient peaks in.
/// Index of the unordered MO pair `{p, q}` in the **packed** (lower-triangle,
/// column-major) pair layout produced by [`pack_pairs_in_place`] and read by
/// [`GwGradEngine::fold_covector_packed`] and its streamed variant.
///
/// The layout is `for col { for row in col..nmo { … } }`, i.e. block `col`
/// starts at `col*nmo - col*(col-1)/2`.
#[inline]
pub fn packed_pair_index(nmo: usize, p: usize, q: usize) -> usize {
    let (col, row) = if p <= q { (p, q) } else { (q, p) };
    col * nmo - (col * col.wrapping_sub(1)) / 2 + (row - col)
}

/// Number of pairs in the packed layout of an `nmo`-dimensional pair space.
#[inline]
pub fn packed_pair_count(nmo: usize) -> usize {
    nmo * (nmo + 1) / 2
}

/// Number of packed pairs whose **smaller** index is `< nrow` (the *band*).
///
/// [`packed_pair_index`] is left unchanged, so the banded layout is exactly the *prefix* of the
/// packed one: index `i < banded_pair_count(nmo, nrow)` describes the same pair in both.  A
/// writer therefore keeps its index formula, and a consumer that zips a banded covector against
/// a full-pair one still reads the matching entries — the omitted ones are zero by construction.
///
/// The LR pair covector of the GW/BSE gradient is nonzero only inside the band
/// `nrow = (largest orbital with a QP weight) + 1`: measured at polyene n=3/def2-TZVP,
/// 7194 of 27495 pair planes carry a nonzero value and the largest smaller index is 32, i.e.
/// `nocc + nvir - 1 = 22 + 11 - 1`.  At polyene n=6/def2-TZVPP the band is 32818 of 161596
/// pairs, a 4.9x smaller covector.
#[inline]
pub fn banded_pair_count(nmo: usize, nrow: usize) -> usize {
    let nrow = nrow.min(nmo);
    if nrow == 0 {
        return 0;
    }
    nrow * nmo - nrow * (nrow - 1) / 2
}

/// Expand a packed covector into the full symmetric `[nmo, nmo] x naux` layout
/// (the same value on both entries of each pair), i.e. the inverse of
/// [`pack_pairs_in_place`].  Provided for the legacy full-buffer contractions.
pub fn unpack_pairs(packed: &[f64], nmo: usize, naux: usize) -> Vec<f64> {
    unpack_pairs_banded(packed, nmo, naux, nmo)
}

/// [`unpack_pairs`] for a **banded** packed covector (see [`banded_pair_count`]); the pairs
/// outside the band are zero in the unpacked buffer, exactly as they are in the stored one.
pub fn unpack_pairs_banded(
    packed: &[f64],
    nmo: usize,
    naux: usize,
    nrow_pair: usize,
) -> Vec<f64> {
    let nrow = nrow_pair.max(1).min(nmo);
    assert_eq!(packed.len(), naux * banded_pair_count(nmo, nrow), "unpack_pairs: size mismatch");
    let mut v = vec![0.0f64; naux * nmo * nmo];
    let mut tp = 0usize;
    for col in 0..nrow {
        for row in col..nmo {
            let src = tp * naux;
            let dst = (row + col * nmo) * naux;
            v[dst..dst + naux].copy_from_slice(&packed[src..src + naux]);
            if row != col {
                let dst_t = (col + row * nmo) * naux;
                v[dst_t..dst_t + naux].copy_from_slice(&packed[src..src + naux]);
            }
            tp += 1;
        }
    }
    v
}

/// Pack a pair-symmetric full covector `[nmo, nmo] x naux` (aux fastest) into
/// its lower triangle `[nmo*(nmo+1)/2, naux]` **in place**, returning the packed
/// pair count.  The packed buffer holds the same value the full buffer stored at
/// each pair, so every consumer of the packed form
/// ([`GwGradEngine::fold_covector_packed`] and its streamed variant) sees exactly
/// the same covector while half the memory is resident.
pub fn pack_pairs_in_place(v: &mut Vec<f64>, nmo: usize, naux: usize) -> usize {
    let mut tp = 0usize;
    for col in 0..nmo {
        for row in col..nmo {
            let src = (row + col * nmo) * naux;
            let dst = tp * naux;
            if dst != src {
                v.copy_within(src..src + naux, dst);
            }
            tp += 1;
        }
    }
    v.truncate(tp * naux);
    v.shrink_to_fit();
    tp
}

fn symmetrise_pairs(v: &mut [f64], naux: usize, nmo: usize) {
    for pi in 0..nmo {
        for pj in (pi + 1)..nmo {
            let a = (pi + pj * nmo) * naux;
            let b = (pj + pi * nmo) * naux;
            for p in 0..naux {
                let s = 0.5 * (v[a + p] + v[b + p]);
                v[a + p] = s;
                v[b + p] = s;
            }
        }
    }
}

fn symmetrise_square(v: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let a = i + j * n;
            let b = j + i * n;
            let s = 0.5 * (v[a] + v[b]);
            v[a] = s;
            v[b] = s;
        }
    }
}

