/// TDDFT A and B matrix-vector product implementations
///
/// Implements the full TDDFT linear response A and B matrix-vector products:
///
/// (A·z)_{ia} = (ε_a - ε_i) * z_{ia}
///              + 2 * Σ_{jb} (ia|jb) * z_{jb}           [Coulomb, singlet]
///              - c_x * Σ_{jb} (ij|ab) * z_{jb}         [Exchange, hybrid only]
///              + fxc_{ia,jb} * z_{jb}                  [XC kernel]
///
/// (B·z)_{ia} = 2 * Σ_{jb} (ia|jb) * z_{jb}             [Coulomb, singlet]
///              - c_x * Σ_{jb} (ja|ib) * z_{jb}         [Exchange, hybrid only]
///              + fxc_{ia,jb} * z_{jb}                  [XC kernel]
///
/// The Coulomb term is computed via RI: (ia|jb) = Σ_Q (ia|Q) * (Q|jb)
/// The Exchange term is computed via RI: (ij|ab) = Σ_Q (ij|Q) * (Q|ab)

use rest_tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dgemv};
use crate::scf_io::SCF;
use crate::ri_bse;
use crate::dft::num_int::{fxc_matvec, fxc_matvec_unrestricted};
use crate::ri_tddft::TDDFTData;

// ── Timing instrumentation for the MO matvec (debug level; see `mo_timing_report`) ──
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crate::ri_tddft::matvec_ao::{add_ns, s_of};

pub static T_MV_J: AtomicU64 = AtomicU64::new(0); // Coulomb (RI contraction)
pub static T_MV_K: AtomicU64 = AtomicU64::new(0); // Exchange (RI tensor DGEMM chain)
pub static T_MV_FXC: AtomicU64 = AtomicU64::new(0); // fxc kernel application
pub static T_MV_ALL: AtomicU64 = AtomicU64::new(0); // whole matvec (incl. diagonal)
pub static N_MO_MV: AtomicU64 = AtomicU64::new(0);

/// Print the accumulated MO-matvec timing table (debug level, visible with
/// print_level >= 2). Mirrors `matvec_ao::ao_timing_report`. Note: the MO
/// Davidson applies the matvec per trial vector, so call counts are per vector.
pub fn mo_timing_report() {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    let n = N_MO_MV.load(Ordering::Relaxed);
    let mv = s_of(&T_MV_ALL);
    log::debug!("MO matvec timing ({} per-vector calls, total {:.3} s):", n, mv);
    for (name, t) in [
        ("J (Coulomb)", &T_MV_J),
        ("K (exchange)", &T_MV_K),
        ("fxc total", &T_MV_FXC),
    ] {
        let v = s_of(t);
        let pct = if mv > 0.0 { 100.0 * v / mv } else { 0.0 };
        log::debug!("  {:<12} {:>10.3} s  ({:>5.1}% of matvec)", name, v, pct);
    }
}

/// Per-sector MO-basis RI tensors of the TDDFT response (MO mode): the
/// Coulomb tensor plus the reshaped HF/RSH exchange tensors, bundled so that
/// one sector loop serves restricted (one sector) and unrestricted
/// (alpha/beta) references alike.
///
/// The exchange contribution of A and B reads
/// `-coeff_full * K_full[z] - coeff_sr * K_SR[z]`, where `K_full` is built
/// from the full-range 3-center integrals and `K_SR` from the short-range
/// (erfc(omega*r12)/r12) integrals. The coefficients themselves are NOT
/// stored here: they derive from `scf.mol.xc_data` (`rsh_params()` +
/// `dfa_hybrid_scf`) at each use site.
///
/// For ordinary hybrids (and pure functionals) the SR tensors are absent and
/// the exchange is the full-range term alone, scaled by the hybrid
/// coefficient. For a range-separated hybrid the HF exchange is
/// `c_SR*K_SR + c_LR*K_LR`, evaluated as
/// `coeff_full*K_full + coeff_sr*K_SR` with `coeff_full = c_LR` and
/// `coeff_sr = c_SR - c_LR` (since `K_full = K_SR + K_LR`), mirroring the
/// ground-state Fock build in `scf_io`.
pub struct RITensorTerms {
    /// Sector window this bundle was built for.
    pub occ_size: usize,
    pub vir_size: usize,
    /// [naux, occ*vir], Coulomb (Hartree) contraction.
    pub coulomb: MatrixFull<f64>,
    /// [occ*naux, occ], A-block exchange.
    pub oo_exch: MatrixFull<f64>,
    /// [naux*vir, vir], A-block exchange.
    pub vv_exch: MatrixFull<f64>,
    /// [naux*occ, vir], B-block exchange.
    pub ov_exch: MatrixFull<f64>,
    /// Short-range (erfc) exchange tensors; `None` for a non-RSH DFA. All
    /// three are set together or none are (all-or-nothing SR triple).
    pub oo_sr: Option<MatrixFull<f64>>,
    pub vv_sr: Option<MatrixFull<f64>>,
    pub ov_sr: Option<MatrixFull<f64>>,
}

/// Build the diagonal preconditioner from KS orbital energy differences:
/// per sector, `hdiag[i + a*occ] = ε_{lumo+a} − ε_{start_mo+i}`, the sectors
/// concatenated as `[alpha (dim_a); beta (dim_b)]` for UHF (one sector for
/// RHF/ROHF), matching the amplitude layout of the corresponding matvecs.
///
/// Uses KS eigenvalues from scf.eigenvalues, NOT GW quasiparticle energies.
pub fn build_hdiag(scf: &SCF) -> Vec<f64> {
    let sectors = crate::ri_tddft::utils::tddft_sector_params(scf);
    let mut hdiag = Vec::new();
    for (i_spin, sec) in sectors.iter().enumerate() {
        let ks = &scf.eigenvalues[i_spin];
        for a in 0..sec.vir_size {
            for i in 0..sec.occ_size {
                hdiag.push(ks[sec.lumo + a] - ks[sec.start_mo + i]);
            }
        }
    }
    hdiag
}

/// A-block exchange kernel: K_A = -alpha * RI_OO^T · (RI_VV · z^T)^T
///
/// Follows the same DGEMM pattern as ri_bse::matvec::w_contribution_a_block_dgemm
/// but uses raw RI_OO/RI_VV instead of screened integrals.
///
/// ri_oo_reshaped: [occ*naux, occ] (pre-transposed via transpose_and_drop + reshape)
/// ri_vv_reshaped: [naux*vir, vir]
/// z: [occ*vir]
/// Returns: [occ*vir] vector, K_A · z with sign -alpha applied
pub fn exchange_a_matvec(
    ri_oo_reshaped: &MatrixFull<f64>,
    ri_vv_reshaped: &MatrixFull<f64>,
    z: &[f64],
    occ_size: usize,
    vir_size: usize,
    alpha_hybrid: f64,
) -> Vec<f64> {
    if alpha_hybrid.abs() < 1e-15 {
        return vec![0.0; occ_size * vir_size];
    }
    let num_auxbas = ri_vv_reshaped.size[0] / vir_size;

    // Step 1: t_tensor[P*vir + a, j] = Σ_b ri_vv[P*vir + a, b] * z_mat^T[b, j]
    // ri_vv_reshaped: [naux*vir, vir]
    // z_mat: [occ, vir], z_mat^T: [vir, occ]
    // t_tensor: [naux*vir, occ]
    let z_mat = MatrixFull::from_vec([occ_size, vir_size], z.to_vec()).unwrap();
    let mut t_tensor = MatrixFull::new([num_auxbas * vir_size, occ_size], 0.0);
    _dgemm_full(ri_vv_reshaped, 'N', &z_mat, 'T', &mut t_tensor, 1.0, 0.0);

    // Step 2: Transpose and reshape
    // t_tensor: [naux*vir, occ] → transpose → [occ, naux*vir]
    // → reshape → [occ*naux, vir]
    t_tensor = t_tensor.transpose_and_drop();
    t_tensor.reshape([num_auxbas * occ_size, vir_size]);

    // Step 3: result[i, a] = -alpha * Σ_j,Σ_P ri_oo[P*nocc + i, j] * t_tensor[j*naux + P, a]
    // ri_oo_reshaped: [occ*naux, occ]
    // t_tensor: [naux*occ, vir]
    // result: [occ, vir]
    let mut result = MatrixFull::new([occ_size, vir_size], 0.0);
    _dgemm_full(ri_oo_reshaped, 'T', &t_tensor, 'N', &mut result, -alpha_hybrid, 0.0);

    result.data
}

/// B-block exchange kernel: K_B = -alpha * (RI_OV · z^T)^T · RI_OV
///
/// Follows the same pattern as ri_bse::matvec::w_contribution_b_block_dgemm
/// but uses raw RI_OV on both sides (no screening).
///
/// ri_ov_reshaped: [naux*occ, vir]
/// z: [occ*vir]
/// Returns: [occ*vir] vector, K_B · z with sign -alpha applied
pub fn exchange_b_matvec(
    ri_ov_reshaped: &MatrixFull<f64>,
    z: &[f64],
    occ_size: usize,
    vir_size: usize,
    alpha_hybrid: f64,
) -> Vec<f64> {
    if alpha_hybrid.abs() < 1e-15 {
        return vec![0.0; occ_size * vir_size];
    }
    let num_auxbas = ri_ov_reshaped.size[0] / occ_size;

    // Step 1: t_tensor[P*nocc + j, i] = Σ_b ri_ov[P*nocc + j, b] * z_mat^T[b, i]
    // ri_ov_reshaped: [naux*occ, vir]
    // z_mat^T: [vir, occ]
    // t_tensor: [naux*occ, occ]
    let z_mat = MatrixFull::from_vec([occ_size, vir_size], z.to_vec()).unwrap();
    let mut t_tensor = MatrixFull::new([num_auxbas * occ_size, occ_size], 0.0);
    _dgemm_full(ri_ov_reshaped, 'N', &z_mat, 'T', &mut t_tensor, 1.0, 0.0);

    // Step 2: Block-swap to convert RI index ordering
    // t_tensor[P*occ + j, i] → t_tensor[P*occ + i, j]
    // This swaps the two occupied indices in the block structure:
    // original block at position (j,i) moves to position (i,j)
    let mut swapped_data = vec![0.0; t_tensor.data.len()];
    for new_idx in 0..occ_size * occ_size {
        let n2 = new_idx / occ_size;
        let n1 = new_idx % occ_size;
        let orig_idx = n1 * occ_size + n2;
        let source_start = orig_idx * num_auxbas;
        let source_end = source_start + num_auxbas;
        swapped_data[new_idx * num_auxbas..(new_idx + 1) * num_auxbas]
            .copy_from_slice(&t_tensor.data[source_start..source_end]);
    }
    t_tensor = MatrixFull::from_vec([num_auxbas * occ_size, occ_size], swapped_data).unwrap();

    // Step 3: result[i, a] = -alpha * Σ_j Σ_P t_tensor^T[i, j*naux+P] * ri_ov[P*nocc + j, a]
    // t_tensor after block-swap: [naux*occ, occ], with t_tensor[P*occ + i, j] = Σ_b (P|jb) * z_{ib}
    // t_tensor^T: [occ, naux*occ]
    // ri_ov_reshaped: [naux*occ, vir]
    // result: [occ, vir]
    let mut result = MatrixFull::new([occ_size, vir_size], 0.0);
    _dgemm_full(&t_tensor, 'T', ri_ov_reshaped, 'N', &mut result, -alpha_hybrid, 0.0);

    result.data
}

/// Full A-block matrix-vector product for TDDFT, sector-generic: one call
/// serves the restricted single-sector reference (z = [occ*vir]) and the
/// unrestricted concatenated reference (z = [z_alpha (dim_a); z_beta (dim_b)]).
///
/// A·z = (ε_a - ε_i)*z + w_J*J[Σ_τ z^τ] - coeff_full*K_A[z] - coeff_sr*K_SR[z] + fxc[z]
///
/// The diagonal, bare exchange and fxc act within a spin sector; the Coulomb
/// (Hartree) term responds to the summed transition density of all sectors
/// and is the only alpha/beta coupling of an unrestricted reference.
/// Restricted spin adaptation enters as the Coulomb weight: singlet
/// (xlet='S') 2, 'R' 1, triplet (xlet='T') 0; unrestricted uses unit weight.
///
/// For TDA: this is the complete matvec (A matrix only).
/// For full LR: this is the A-block of [A B; -B -A].
pub fn a_matvec(
    scf: &SCF,
    data: &crate::ri_tddft::TDDFTData,
    z: &Vec<f64>,
    xlet: char,
) -> Vec<f64> {
    let t_mv = Instant::now();
    N_MO_MV.fetch_add(1, Ordering::Relaxed);
    let terms = &data.ri_terms;
    let n_sec = terms.len();
    debug_assert_eq!(n_sec, data.n_sectors());
    let dims: Vec<usize> = terms.iter().map(|t| t.occ_size * t.vir_size).collect();
    let dim_total: usize = dims.iter().sum();
    debug_assert_eq!(z.len(), dim_total);

    // Response HF-exchange coefficients from the DFA
    // (RSH -> (c_LR, c_SR - c_LR); hybrid -> (c_x, 0)); the coefficients are
    // spin-independent, derived once here and applied per sector.
    let (coeff_full, coeff_sr) = match scf.mol.xc_data.rsh_params() {
        Some((_, c_lr, c_sr)) => (c_lr, c_sr - c_lr),
        None => (scf.mol.xc_data.dfa_hybrid_scf, 0.0),
    };
    // Coulomb weight: spin-adapted restricted channels vs unit-weight unrestricted.
    let coulomb_factor = if data.is_uhf() {
        1.0
    } else if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    // fxc: the restricted table evaluates the single sector; the
    // spin-resolved table evaluates the whole concatenated block at once
    // and is split per sector below.
    let t_fxc = Instant::now();
    let fxc_parts: Vec<Vec<f64>> = if let Some(fxc_u) = data.fxc_u.as_ref() {
        let (fa, fb) = fxc_matvec_unrestricted(fxc_u, z);
        vec![fa, fb]
    } else {
        let fxc_data = data.fxc.as_ref().expect("MO mode requires fxc data");
        vec![fxc_matvec(fxc_data, &z[..dims[0]])]
    };
    add_ns(&T_MV_FXC, t_fxc);

    // Sector windows for the diagonal.
    let sectors = crate::ri_tddft::utils::tddft_sector_params(scf);
    debug_assert_eq!(sectors.len(), n_sec);

    let mut result = vec![0.0; dim_total];
    let mut offset = 0usize;
    for s in 0..n_sec {
        let ns = dims[s];
        if ns == 0 { continue; }
        let zs = &z[offset..offset + ns];
        let mut rs = vec![0.0; ns];
        let term_s = &terms[s];

        // Step 1: Diagonal contribution: (ε_a - ε_i) * z
        let sec = &sectors[s];
        let ks = &scf.eigenvalues[s];
        for a in 0..term_s.vir_size {
            for i in 0..term_s.occ_size {
                let idx = i + a * term_s.occ_size;
                rs[idx] = (ks[sec.lumo + a] - ks[sec.start_mo + i]) * zs[idx];
            }
        }

        // Step 2: Coulomb contribution: w_J * Σ_τ J_{sτ}[z^τ]
        // (restricted: one sector; unrestricted: same-spin + cross-spin sums)
        if coulomb_factor != 0.0 {
            let t0 = Instant::now();
            let mut vs = vec![0.0; ns];
            let mut offset_t = 0usize;
            for t in 0..n_sec {
                if dims[t] == 0 { continue; }
                let zt = z[offset_t..offset_t + dims[t]].to_vec();
                let jt = if s == t {
                    ri_bse::matvec::coulomb_contribution(&term_s.coulomb, &zt)
                } else {
                    ri_bse::matvec::coulomb_cross_contribution(&term_s.coulomb, &terms[t].coulomb, &zt)
                };
                for (v, c) in vs.iter_mut().zip(jt) { *v += c; }
                offset_t += dims[t];
            }
            add_ns(&T_MV_J, t0);
            for idx in 0..ns {
                rs[idx] += coulomb_factor * vs[idx];
            }
        }

        // Step 3: Exchange contribution: -coeff_full * K_A[z] - coeff_sr * K_SR[z]
        if coeff_full.abs() > 1e-15 {
            let t0 = Instant::now();
            let kz = exchange_a_matvec(&term_s.oo_exch, &term_s.vv_exch, zs, term_s.occ_size, term_s.vir_size, coeff_full);
            add_ns(&T_MV_K, t0);
            for idx in 0..ns {
                rs[idx] += kz[idx];
            }
        }
        if coeff_sr.abs() > 1e-15 {
            let (oo_sr, vv_sr) = match (term_s.oo_sr.as_ref(), term_s.vv_sr.as_ref()) {
                (Some(oo), Some(vv)) => (oo, vv),
                _ => panic!("RSH short-range exchange requested (coeff_sr = {}) but the SR exchange tensors are missing", coeff_sr),
            };
            let t0 = Instant::now();
            let kz = exchange_a_matvec(oo_sr, vv_sr, zs, term_s.occ_size, term_s.vir_size, coeff_sr);
            add_ns(&T_MV_K, t0);
            for idx in 0..ns {
                rs[idx] += kz[idx];
            }
        }

        // Step 4: XC kernel contribution: fxc[z]
        for idx in 0..ns {
            rs[idx] += fxc_parts[s][idx];
        }

        result[offset..offset + ns].copy_from_slice(&rs);
        offset += ns;
    }

    add_ns(&T_MV_ALL, t_mv);
    result
}

/// Full B-block matrix-vector product for TDDFT (used in full LR, not TDA),
/// sector-generic (see `a_matvec` for the sector convention).
///
/// B·z = w_J*J[Σ_τ z^τ] - coeff_full*K_B[z] - coeff_sr*K_SR[z] + fxc[z]
///
/// Note: B has no diagonal term (no orbital energy differences).
pub fn b_matvec(
    scf: &SCF,
    data: &crate::ri_tddft::TDDFTData,
    z: &Vec<f64>,
    xlet: char,
) -> Vec<f64> {
    let t_mv = Instant::now();
    N_MO_MV.fetch_add(1, Ordering::Relaxed);
    let terms = &data.ri_terms;
    let n_sec = terms.len();
    debug_assert_eq!(n_sec, data.n_sectors());
    let dims: Vec<usize> = terms.iter().map(|t| t.occ_size * t.vir_size).collect();
    let dim_total: usize = dims.iter().sum();
    debug_assert_eq!(z.len(), dim_total);

    // Response HF-exchange coefficients from the DFA (see a_matvec).
    let (coeff_full, coeff_sr) = match scf.mol.xc_data.rsh_params() {
        Some((_, c_lr, c_sr)) => (c_lr, c_sr - c_lr),
        None => (scf.mol.xc_data.dfa_hybrid_scf, 0.0),
    };
    // Coulomb weight: spin-adapted restricted channels vs unit-weight unrestricted.
    let coulomb_factor = if data.is_uhf() {
        1.0
    } else if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    // fxc (see a_matvec).
    let t_fxc = Instant::now();
    let fxc_parts: Vec<Vec<f64>> = if let Some(fxc_u) = data.fxc_u.as_ref() {
        let (fa, fb) = fxc_matvec_unrestricted(fxc_u, z);
        vec![fa, fb]
    } else {
        let fxc_data = data.fxc.as_ref().expect("MO mode requires fxc data");
        vec![fxc_matvec(fxc_data, &z[..dims[0]])]
    };
    add_ns(&T_MV_FXC, t_fxc);

    let mut result = vec![0.0; dim_total];
    let mut offset = 0usize;
    for s in 0..n_sec {
        let ns = dims[s];
        if ns == 0 { continue; }
        let zs = &z[offset..offset + ns];
        let mut rs = vec![0.0; ns];
        let term_s = &terms[s];

        // Step 1: Coulomb contribution: w_J * Σ_τ J_{sτ}[z^τ]
        if coulomb_factor != 0.0 {
            let t0 = Instant::now();
            let mut vs = vec![0.0; ns];
            let mut offset_t = 0usize;
            for t in 0..n_sec {
                if dims[t] == 0 { continue; }
                let zt = z[offset_t..offset_t + dims[t]].to_vec();
                let jt = if s == t {
                    ri_bse::matvec::coulomb_contribution(&term_s.coulomb, &zt)
                } else {
                    ri_bse::matvec::coulomb_cross_contribution(&term_s.coulomb, &terms[t].coulomb, &zt)
                };
                for (v, c) in vs.iter_mut().zip(jt) { *v += c; }
                offset_t += dims[t];
            }
            add_ns(&T_MV_J, t0);
            for idx in 0..ns {
                rs[idx] += coulomb_factor * vs[idx];
            }
        }

        // Step 2: Exchange contribution: -coeff_full * K_B[z] - coeff_sr * K_SR[z]
        if coeff_full.abs() > 1e-15 {
            let t0 = Instant::now();
            let kz = exchange_b_matvec(&term_s.ov_exch, zs, term_s.occ_size, term_s.vir_size, coeff_full);
            add_ns(&T_MV_K, t0);
            for idx in 0..ns {
                rs[idx] += kz[idx];
            }
        }
        if coeff_sr.abs() > 1e-15 {
            let ov_sr = match term_s.ov_sr.as_ref() {
                Some(ov) => ov,
                None => panic!("RSH short-range exchange requested (coeff_sr = {}) but the SR exchange tensors are missing", coeff_sr),
            };
            let t0 = Instant::now();
            let kz = exchange_b_matvec(ov_sr, zs, term_s.occ_size, term_s.vir_size, coeff_sr);
            add_ns(&T_MV_K, t0);
            for idx in 0..ns {
                rs[idx] += kz[idx];
            }
        }

        // Step 3: XC kernel contribution: fxc[z]
        for idx in 0..ns {
            rs[idx] += fxc_parts[s][idx];
        }

        result[offset..offset + ns].copy_from_slice(&rs);
        offset += ns;
    }

    add_ns(&T_MV_ALL, t_mv);
    result
}


// ====== Tests with synthetic data ======

#[cfg(test)]
mod tests {
    use super::*;

    fn build_ri_matrices(occ_size: usize, vir_size: usize, naux: usize) -> (MatrixFull<f64>, MatrixFull<f64>, MatrixFull<f64>) {
        // RI_OV: [naux, occ*vir]
        let mut ri_ov_data = Vec::with_capacity(naux * occ_size * vir_size);
        for P in 0..naux {
            for i in 0..occ_size {
                for a in 0..vir_size {
                    ri_ov_data.push(((P + 1) as f64 * (i + 1) as f64 * (a + 1) as f64).sin() * 0.1);
                }
            }
        }
        let ri_ov = MatrixFull::from_vec([naux, occ_size * vir_size], ri_ov_data).unwrap();

        // RI_OO: [naux, occ*occ]
        let mut ri_oo_data = Vec::with_capacity(naux * occ_size * occ_size);
        for P in 0..naux {
            for i in 0..occ_size {
                for j in 0..occ_size {
                    ri_oo_data.push(((P + 1) as f64 * (i + 1) as f64 * (j + 1) as f64).cos() * 0.05);
                }
            }
        }
        let ri_oo = MatrixFull::from_vec([naux, occ_size * occ_size], ri_oo_data).unwrap();

        // RI_VV: [naux, vir*vir]
        let mut ri_vv_data = Vec::with_capacity(naux * vir_size * vir_size);
        for P in 0..naux {
            for a in 0..vir_size {
                for b in 0..vir_size {
                    ri_vv_data.push(((P + 1) as f64 * (a + 1) as f64 * (b + 1) as f64).sin() * 0.05);
                }
            }
        }
        let ri_vv = MatrixFull::from_vec([naux, vir_size * vir_size], ri_vv_data).unwrap();

        (ri_ov, ri_oo, ri_vv)
    }

    fn build_exch_matrices(ri_oo: &MatrixFull<f64>, ri_vv: &MatrixFull<f64>, ri_ov: &MatrixFull<f64>,
                           occ_size: usize, vir_size: usize, naux: usize)
        -> (MatrixFull<f64>, MatrixFull<f64>, MatrixFull<f64>)
    {
        // ri_oo → [occ*naux, occ]
        let mut ri_oo_exch = ri_oo.clone();
        ri_oo_exch.reshape([naux * occ_size, occ_size]);
        ri_oo_exch = ri_oo_exch.transpose_and_drop();
        ri_oo_exch.reshape([occ_size * naux, occ_size]);

        // ri_vv → [naux*vir, vir]
        let mut ri_vv_exch = ri_vv.clone();
        ri_vv_exch.reshape([naux * vir_size, vir_size]);

        // ri_ov → [naux*occ, vir]
        let mut ri_ov_exch = ri_ov.clone();
        ri_ov_exch.reshape([naux * occ_size, vir_size]);

        (ri_oo_exch, ri_vv_exch, ri_ov_exch)
    }

    #[test]
    fn test_hdiag() {
        // hdiag is simplest - just check it returns the right length
        // We can't test values without SCF, so just verify the function compiles
    }

    #[test]
    fn test_exchange_a_matvec_shape() {
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, ri_oo, ri_vv) = build_ri_matrices(occ, vir, naux);
        let (ri_oo_exch, ri_vv_exch, _) = build_exch_matrices(&ri_oo, &ri_vv, &ri_ov, occ, vir, naux);
        let n = occ * vir;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let result = exchange_a_matvec(&ri_oo_exch, &ri_vv_exch, &z, occ, vir, 0.25);
        assert_eq!(result.len(), n);
        assert!(result.iter().all(|x| x.is_finite()));
        println!("exchange_a_matvec shape OK, first 3: {:?}", &result[..3]);
    }

    #[test]
    fn test_exchange_a_matvec_zero_hybrid() {
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, ri_oo, ri_vv) = build_ri_matrices(occ, vir, naux);
        let (ri_oo_exch, ri_vv_exch, _) = build_exch_matrices(&ri_oo, &ri_vv, &ri_ov, occ, vir, naux);
        let n = occ * vir;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let result = exchange_a_matvec(&ri_oo_exch, &ri_vv_exch, &z, occ, vir, 0.0);
        assert!(result.iter().all(|x| x.abs() < 1e-15));
    }

    #[test]
    fn test_exchange_a_matvec_zero_z() {
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, ri_oo, ri_vv) = build_ri_matrices(occ, vir, naux);
        let (ri_oo_exch, ri_vv_exch, _) = build_exch_matrices(&ri_oo, &ri_vv, &ri_ov, occ, vir, naux);
        let n = occ * vir;
        let z = vec![0.0; n];
        let result = exchange_a_matvec(&ri_oo_exch, &ri_vv_exch, &z, occ, vir, 0.25);
        assert!(result.iter().all(|x| x.abs() < 1e-15));
    }

    #[test]
    fn test_exchange_b_matvec_shape() {
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, ri_oo, ri_vv) = build_ri_matrices(occ, vir, naux);
        let (_, _, ri_ov_exch) = build_exch_matrices(&ri_oo, &ri_vv, &ri_ov, occ, vir, naux);
        let n = occ * vir;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let result = exchange_b_matvec(&ri_ov_exch, &z, occ, vir, 0.25);
        assert_eq!(result.len(), n);
        assert!(result.iter().all(|x| x.is_finite()));
        println!("exchange_b_matvec shape OK, first 3: {:?}", &result[..3]);
    }

    #[test]
    fn test_exchange_b_matvec_zero_z() {
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, ri_oo, ri_vv) = build_ri_matrices(occ, vir, naux);
        let (_, _, ri_ov_exch) = build_exch_matrices(&ri_oo, &ri_vv, &ri_ov, occ, vir, naux);
        let z = vec![0.0; occ * vir];
        let result = exchange_b_matvec(&ri_ov_exch, &z, occ, vir, 0.25);
        assert!(result.iter().all(|x| x.abs() < 1e-15));
    }

    #[test]
    fn test_coulomb_contribution_symmetry() {
        // Test that J^T = J (Coulomb is symmetric)
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, _, _) = build_ri_matrices(occ, vir, naux);
        let n = occ * vir;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let z2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let r1 = ri_bse::matvec::coulomb_contribution(&ri_ov, &z1);
        let r2 = ri_bse::matvec::coulomb_contribution(&ri_ov, &z2);
        let v12: f64 = z1.iter().zip(r2.iter()).map(|(a, b)| a * b).sum();
        let v21: f64 = z2.iter().zip(r1.iter()).map(|(a, b)| a * b).sum();
        assert!((v12 - v21).abs() < 1e-14, "Coulomb matvec not symmetric: {} vs {}", v12, v21);
        println!("Coulomb symmetry: ⟨z1,J·z2⟩ = {:.10}, ⟨z2,J·z1⟩ = {:.10}", v12, v21);
    }

    #[test]
    fn test_exchange_a_matvec_random() {
        let occ = 3; let vir = 5; let naux = 4;
        let (ri_ov, ri_oo, ri_vv) = build_ri_matrices(occ, vir, naux);
        let (ri_oo_exch, ri_vv_exch, _) = build_exch_matrices(&ri_oo, &ri_vv, &ri_ov, occ, vir, naux);
        let n = occ * vir;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let result_a = exchange_a_matvec(&ri_oo_exch, &ri_vv_exch, &z, occ, vir, 0.25);
        assert!(result_a.len() == n);
        assert!(result_a.iter().all(|x| x.is_finite()));
    }
}
