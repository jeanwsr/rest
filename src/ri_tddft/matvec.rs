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
use crate::dft::num_int::{FXCMatvecData, FXCMatvecDataUnrestricted, fxc_matvec, fxc_matvec_unrestricted};
use crate::ri_tddft::utils::tddft_occupation_parameters;

/// Build the diagonal preconditioner from KS orbital energy differences
///
/// hdiag[i + a*nocc] = ε_{lumo+a} - ε_{start_mo+i}
///
/// Uses KS eigenvalues from scf.eigenvalues[0], NOT GW quasiparticle energies.
pub fn build_hdiag(scf: &SCF) -> Vec<f64> {
    let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
        tddft_occupation_parameters(scf);
    let ks = &scf.eigenvalues[0];
    let mut hdiag = Vec::with_capacity(occ_size * vir_size);
    for a in 0..vir_size {
        for i in 0..occ_size {
            hdiag.push(ks[lumo + a] - ks[start_mo + i]);
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

/// Full A-block matrix-vector product for TDDFT
///
/// A·z = (ε_a - ε_i)*z + 2*Σ_jb(ia|jb)*z_jb - c_x*Σ_jb(ij|ab)*z_jb + fxc[z]
///
/// For TDA: this is the complete matvec (A matrix only).
/// For full LR: this is the A-block of [A B; -B -A].
///
/// singlet (xlet='S'): Coulomb factor = 2
/// triplet (xlet='T'): Coulomb factor = 0
pub fn a_matvec(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,          // [naux, occ*vir], for Coulomb
    ri_oo_exch: &MatrixFull<f64>,     // [occ*naux, occ], for A exchange
    ri_vv_exch: &MatrixFull<f64>,     // [naux*vir, vir], for A exchange
    z: &Vec<f64>,
    xlet: char,
    alpha_hybrid: f64,
) -> Vec<f64> {
    let occ_size = fxc_data.nocc;
    let vir_size = fxc_data.nvir;
    let dim = occ_size * vir_size;

    // Build diagonal using KS eigenvalues
    let (start_mo, _num_state, _, _, _homo, lumo) = tddft_occupation_parameters(scf);
    let ks = &scf.eigenvalues[0];

    // Step 1: Diagonal contribution: (ε_a - ε_i) * z
    let mut result = vec![0.0; dim];
    for a in 0..vir_size {
        for i in 0..occ_size {
            let idx = i + a * occ_size;
            result[idx] = (ks[lumo + a] - ks[start_mo + i]) * z[idx];
        }
    }

    // Step 2: Coulomb contribution: 2 * J[z] (singlet only)
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };
    if coulomb_factor != 0.0 {
        let jz = ri_bse::matvec::coulomb_contribution(ri_ov, z);
        for idx in 0..dim {
            result[idx] += coulomb_factor * jz[idx];
        }
    }

    // Step 3: Exchange contribution: -c_x * K_A[z] (hybrid only)
    if alpha_hybrid.abs() > 1e-15 {
        let kz = exchange_a_matvec(ri_oo_exch, ri_vv_exch, z, occ_size, vir_size, alpha_hybrid);
        for idx in 0..dim {
            result[idx] += kz[idx];
        }
    }

    // Step 4: XC kernel contribution: fxc[z]
    let fxc = fxc_matvec(fxc_data, z);
    for idx in 0..dim {
        result[idx] += fxc[idx];
    }

    result
}

/// Full B-block matrix-vector product for TDDFT (used in full LR, not TDA)
///
/// B·z = 2*Σ_jb(ia|jb)*z_jb - c_x*Σ_jb(ja|ib)*z_jb + fxc[z]
///
/// Note: B has no diagonal term (no orbital energy differences).
///
/// singlet (xlet='S'): Coulomb factor = 2
/// triplet (xlet='T'): Coulomb factor = 0
pub fn b_matvec(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,          // [naux, occ*vir], for Coulomb
    ri_ov_exch: &MatrixFull<f64>,     // [naux*occ, vir], for B exchange
    z: &Vec<f64>,
    xlet: char,
    alpha_hybrid: f64,
) -> Vec<f64> {
    let occ_size = fxc_data.nocc;
    let vir_size = fxc_data.nvir;
    let dim = occ_size * vir_size;

    let mut result = vec![0.0; dim];

    // Step 1: Coulomb contribution: 2 * J[z] (singlet only)
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };
    if coulomb_factor != 0.0 {
        let jz = ri_bse::matvec::coulomb_contribution(ri_ov, z);
        for idx in 0..dim {
            result[idx] += coulomb_factor * jz[idx];
        }
    }

    // Step 2: Exchange contribution: -c_x * K_B[z]
    if alpha_hybrid.abs() > 1e-15 {
        let kz = exchange_b_matvec(ri_ov_exch, z, occ_size, vir_size, alpha_hybrid);
        for idx in 0..dim {
            result[idx] += kz[idx];
        }
    }

    // Step 3: XC kernel contribution: fxc[z]
    let fxc = fxc_matvec(fxc_data, z);
    for idx in 0..dim {
        result[idx] += fxc[idx];
    }

    result
}


// ============================================================================
// Unrestricted TDDFT matrix-vector products
// ============================================================================

/// Coulomb contribution for unrestricted TDDFT from every spin channel.
///
/// For output spin `s`:
///   V_s(z) = Σ_t ri_ov[s]^T (ri_ov[t] z_t)
/// with no extra spin-degeneracy factor.
fn unrestricted_coulomb_sum(
    ri_ov: &[MatrixFull<f64>; 2],
    n0: usize,
    n1: usize,
    occ_sizes: [usize; 2],
    vir_sizes: [usize; 2],
    z: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ns = [occ_sizes[0] * vir_sizes[0], occ_sizes[1] * vir_sizes[1]];
    debug_assert_eq!(z.len(), n0 + n1);
    let mut va = vec![0.0; ns[0]];
    let mut vb = vec![0.0; ns[1]];

    if ns[0] > 0 {
        let z0 = z[..n0].to_vec();
        let z1 = z[n0..].to_vec();
        let v00 = ri_bse::matvec::coulomb_contribution(&ri_ov[0], &z0);
        va.iter_mut().zip(v00).for_each(|(a, b)| *a += b);
        if ns[1] > 0 {
            let v01 = ri_bse::matvec::coulomb_cross_contribution(&ri_ov[0], &ri_ov[1], &z1);
            va.iter_mut().zip(v01).for_each(|(a, b)| *a += b);
        }
    }
    if ns[1] > 0 {
        let z0 = z[..n0].to_vec();
        let z1 = z[n0..].to_vec();
        let v10 = ri_bse::matvec::coulomb_cross_contribution(&ri_ov[1], &ri_ov[0], &z0);
        vb.iter_mut().zip(v10).for_each(|(a, b)| *a += b);
        let v11 = ri_bse::matvec::coulomb_contribution(&ri_ov[1], &z1);
        vb.iter_mut().zip(v11).for_each(|(a, b)| *a += b);
    }
    (va, vb)
}

/// Full unrestricted TDDFT A-block matrix-vector product.
///
/// The vector is the concatenation `[alpha; beta]`.  `with_hartree` controls
/// whether the Coulomb coupling is included (the “singlet-like” unrestricted
/// mode).  The same-spin bare exchange and the spin-resolved fxc kernel are
/// always included.
pub fn a_matvec_unrestricted(
    scf: &SCF,
    fxc_data: &FXCMatvecDataUnrestricted,
    ri_ov: &[MatrixFull<f64>; 2],
    ri_oo_exch: &[MatrixFull<f64>; 2],
    ri_vv_exch: &[MatrixFull<f64>; 2],
    z: &[f64],
    with_hartree: bool,
    alpha_hybrid: f64,
) -> Vec<f64> {
    let n0 = fxc_data.nocc[0] * fxc_data.nvir[0];
    let n1 = fxc_data.nocc[1] * fxc_data.nvir[1];
    assert_eq!(z.len(), n0 + n1);
    let mut result = vec![0.0; n0 + n1];

    // Per-spin diagonal, exchange and fxc.
    let (fa, fb) = fxc_matvec_unrestricted(fxc_data, z);
    for s in 0..2 {
        let occ_s = fxc_data.nocc[s];
        let vir_s = fxc_data.nvir[s];
        let ns = occ_s * vir_s;
        if ns == 0 { continue; }
        let offset = if s == 0 { 0 } else { n0 };
        let zs = &z[offset..offset + ns];
        let mut rs = vec![0.0; ns];

        let (start_mo, _num_state, _occ, _vir, _homo, lumo) =
            crate::ri_tddft::utils::tddft_occupation_parameters_spin(scf, s);
        let ks = &scf.eigenvalues[s];
        for a in 0..vir_s {
            for i in 0..occ_s {
                let idx = i + a * occ_s;
                rs[idx] = (ks[lumo + a] - ks[start_mo + i]) * zs[idx];
            }
        }

        if alpha_hybrid.abs() > 1e-15 {
            let kz = exchange_a_matvec(
                &ri_oo_exch[s],
                &ri_vv_exch[s],
                zs,
                occ_s,
                vir_s,
                alpha_hybrid,
            );
            for idx in 0..ns {
                rs[idx] += kz[idx];
            }
        }

        if s == 0 {
            for idx in 0..ns { rs[idx] += fa[idx]; }
        } else {
            for idx in 0..ns { rs[idx] += fb[idx]; }
        }
        result[offset..offset + ns].copy_from_slice(&rs);
    }

    if with_hartree {
        let (va, vb) = unrestricted_coulomb_sum(ri_ov, n0, n1, fxc_data.nocc, fxc_data.nvir, z);
        for idx in 0..n0 { result[idx] += va[idx]; }
        for idx in 0..n1 { result[n0 + idx] += vb[idx]; }
    }

    result
}

/// Full unrestricted TDDFT B-block matrix-vector product.
pub fn b_matvec_unrestricted(
    scf: &SCF,
    fxc_data: &FXCMatvecDataUnrestricted,
    ri_ov: &[MatrixFull<f64>; 2],
    ri_ov_exch: &[MatrixFull<f64>; 2],
    z: &[f64],
    with_hartree: bool,
    alpha_hybrid: f64,
) -> Vec<f64> {
    let n0 = fxc_data.nocc[0] * fxc_data.nvir[0];
    let n1 = fxc_data.nocc[1] * fxc_data.nvir[1];
    assert_eq!(z.len(), n0 + n1);
    let mut result = vec![0.0; n0 + n1];

    let (fa, fb) = fxc_matvec_unrestricted(fxc_data, z);
    for s in 0..2 {
        let occ_s = fxc_data.nocc[s];
        let vir_s = fxc_data.nvir[s];
        let ns = occ_s * vir_s;
        if ns == 0 { continue; }
        let offset = if s == 0 { 0 } else { n0 };
        let zs = &z[offset..offset + ns];
        let mut rs = vec![0.0; ns];

        if alpha_hybrid.abs() > 1e-15 {
            let kz = exchange_b_matvec(
                &ri_ov_exch[s],
                zs,
                occ_s,
                vir_s,
                alpha_hybrid,
            );
            for idx in 0..ns {
                rs[idx] += kz[idx];
            }
        }

        if s == 0 {
            for idx in 0..ns { rs[idx] += fa[idx]; }
        } else {
            for idx in 0..ns { rs[idx] += fb[idx]; }
        }
        result[offset..offset + ns].copy_from_slice(&rs);
    }

    if with_hartree {
        let (va, vb) = unrestricted_coulomb_sum(ri_ov, n0, n1, fxc_data.nocc, fxc_data.nvir, z);
        for idx in 0..n0 { result[idx] += va[idx]; }
        for idx in 0..n1 { result[n0 + idx] += vb[idx]; }
    }

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
