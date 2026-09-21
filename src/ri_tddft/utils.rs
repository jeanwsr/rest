/// TDDFT utility functions (no dependency on quasiparticle_methods)
///
/// Provides self-contained replacements for functions from ri_gw and ri_bse
/// that require QuasiParticle control parameters.

use rest_tensors::{MatrixFull, RIFull};
use crate::scf_io::SCF;

/// Get TDDFT orbital parameters without quasiparticle_methods dependency
///
/// Returns (start_mo, num_state, occ_size, vir_size, homo, lumo)
///
/// Automatically freezes core orbitals based on a simple energy heuristic:
/// orbitals with eigenvalue < FROZEN_CORE_THRESHOLD (in Hartree) are frozen.
/// This prevents unphysical core→virtual transitions from polluting the
/// TDDFT excitation space, which would cause convergence issues in the
/// Davidson solver.
pub fn tddft_occupation_parameters(scf: &SCF) -> (usize, usize, usize, usize, usize, usize) {
    let cutoff = scf.mol.ctrl.tddft.as_ref()
        .map(|c| c.tddft_cutoff_energy)
        .unwrap_or(1.0e6);
    let mut num_state = scf.mol.num_state;
    let mut homo = 0;
    let mut lumo = num_state;
    for i_spin in 0..scf.mol.spin_channel {
        let i_homo = scf.homo[i_spin];
        let i_lumo = scf.lumo[i_spin];
        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }
    // Do not freeze core orbitals automatically.  PySCF's TDDFT also uses all
    // occupied orbitals by default, so this keeps the active spaces identical.
    let start_mo = scf.mol.start_mo;
    let occ_size = homo - start_mo + 1;
    let ks = &scf.eigenvalues[0];
    // Apply virtual orbital energy cutoff (mirrors BSE's get_occupation_parameters)
    if cutoff < 1.0e5 {
        num_state = ks.iter().filter(|x| **x < cutoff).count();
        if num_state < homo + 1 {
            num_state = homo + 1; // keep at least all occupied orbitals
        }
    }
    let vir_size = num_state - lumo;
    (start_mo, num_state, occ_size, vir_size, homo, lumo)
}


/// Get TDDFT orbital parameters for one spin channel (unrestricted).
///
/// Same meaning as [`tddft_occupation_parameters`], but uses the spin-resolved
/// eigenvalues, occupation windows and cutoff.
pub fn tddft_occupation_parameters_spin(
    scf: &SCF,
    spin: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    let cutoff = scf.mol.ctrl.tddft.as_ref()
        .map(|c| c.tddft_cutoff_energy)
        .unwrap_or(1.0e6);
    let mut num_state = scf.mol.num_state;
    let homo = scf.homo[spin];
    let lumo = scf.lumo[spin].min(num_state);
    let ks = &scf.eigenvalues[spin];

    // Do not freeze core orbitals automatically; keep all occupied orbitals to
    // match PySCF's default no-frozen TDDFT.
    let start_mo = scf.mol.start_mo;

    // If there are no occupied orbitals in this spin channel, occ_size is zero.
    let occ_size = (start_mo..=homo)
        .filter(|&i| scf.occupation[spin].get(i).map(|&x| x > 1.0e-6).unwrap_or(false))
        .count();

    if cutoff < 1.0e5 {
        num_state = ks.iter().filter(|x| **x < cutoff).count();
        if num_state < homo + 1 {
            num_state = homo + 1;
        }
    }
    let vir_size = num_state.saturating_sub(lumo);
    (start_mo, num_state, occ_size, vir_size, homo, lumo)
}
/// Get RI submatrix for TDDFT (no quasiparticle_methods dependency)
///
/// Directly calls generate_ri3mo_rayon_for_multiple_times with the appropriate
/// MO ranges computed from the given orbital parameters.
pub fn tddft_get_submatrix(
    scf: &SCF,
    choice_a: char,
    choice_b: char,
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    homo: usize,
    lumo: usize,
    num_state: usize,
) -> MatrixFull<f64> {
    let ranges = tddft_submatrix_ranges(choice_a, choice_b, start_mo, homo, lumo, num_state);
    let vector = scf.generate_ri3mo_rayon_for_multiple_times(ranges.0, ranges.1);
    let matrix: MatrixFull<f64> = vector[0].0.rifull_to_matfull_i_jk();
    println!(
        "TDDFT RI Tensor: {}-{}, Size={:?}, naux={}",
        choice_a, choice_b, matrix.size, matrix.size[0]
    );
    matrix
}

/// Get the short-range (RSH, erfc operator) RI submatrix for TDDFT.
///
/// Same construction as [`tddft_get_submatrix`] but built from the
/// short-range 3-center integrals (`rimatr_sr`).
pub fn tddft_get_submatrix_sr(
    scf: &SCF,
    choice_a: char,
    choice_b: char,
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    homo: usize,
    lumo: usize,
    num_state: usize,
) -> MatrixFull<f64> {
    let ranges = tddft_submatrix_ranges(choice_a, choice_b, start_mo, homo, lumo, num_state);
    let vector = scf.generate_ri3mo_sr_rayon_for_multiple_times(ranges.0, ranges.1);
    let matrix: MatrixFull<f64> = vector[0].0.rifull_to_matfull_i_jk();
    println!(
        "TDDFT RI Tensor (SR): {}-{}, Size={:?}, naux={}",
        choice_a, choice_b, matrix.size, matrix.size[0]
    );
    matrix
}

/// Get a spin-resolved RI submatrix for unrestricted TDDFT.
///
/// The MO range arguments belong to one spin channel.  For an unrestricted
/// SCF, `generate_ri3mo_rayon_for_multiple_times` returns one RI block per
/// spin; this helper picks the requested one.
pub fn tddft_get_submatrix_spin(
    scf: &SCF,
    choice_a: char,
    choice_b: char,
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    homo: usize,
    lumo: usize,
    num_state: usize,
    spin: usize,
) -> MatrixFull<f64> {
    let ranges = tddft_submatrix_ranges(choice_a, choice_b, start_mo, homo, lumo, num_state);
    let vector = scf.generate_ri3mo_rayon_for_multiple_times(ranges.0, ranges.1);
    if spin >= vector.len() {
        panic!(
            "tddft_get_submatrix_spin: spin {} requested but only {} RI channels are available",
            spin,
            vector.len()
        );
    }
    let matrix: MatrixFull<f64> = vector[spin].0.rifull_to_matfull_i_jk();
    println!(
        "TDDFT RI Tensor (spin {}): {}-{}, Size={:?}, naux={}",
        spin, choice_a, choice_b, matrix.size, matrix.size[0]
    );
    matrix
}

/// Get a spin-resolved short-range (RSH) RI submatrix for unrestricted TDDFT.
pub fn tddft_get_submatrix_sr_spin(
    scf: &SCF,
    choice_a: char,
    choice_b: char,
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    homo: usize,
    lumo: usize,
    num_state: usize,
    spin: usize,
) -> MatrixFull<f64> {
    let ranges = tddft_submatrix_ranges(choice_a, choice_b, start_mo, homo, lumo, num_state);
    let vector = scf.generate_ri3mo_sr_rayon_for_multiple_times(ranges.0, ranges.1);
    if spin >= vector.len() {
        panic!(
            "tddft_get_submatrix_sr_spin: spin {} requested but only {} RI channels are available",
            spin,
            vector.len()
        );
    }
    let matrix: MatrixFull<f64> = vector[spin].0.rifull_to_matfull_i_jk();
    println!(
        "TDDFT RI Tensor (SR, spin {}): {}-{}, Size={:?}, naux={}",
        spin, choice_a, choice_b, matrix.size, matrix.size[0]
    );
    matrix
}

/// MO ranges of the occ-occ / vir-vir / occ-vir RI blocks for TDDFT.
fn tddft_submatrix_ranges(
    choice_a: char,
    choice_b: char,
    start_mo: usize,
    homo: usize,
    lumo: usize,
    num_state: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let range_oo = (start_mo..homo + 1, start_mo..homo + 1);
    let range_vv = (lumo..num_state, lumo..num_state);
    let range_ov = (start_mo..homo + 1, lumo..num_state);
    if choice_a == 'O' && choice_b == 'O' {
        range_oo
    } else if choice_a == 'V' && choice_b == 'V' {
        range_vv
    } else if choice_a == 'O' && choice_b == 'V' {
        range_ov
    } else {
        panic!("tddft_submatrix_ranges: invalid choice {}/{}", choice_a, choice_b);
    }
}

/// Reshape the raw [naux, n*m] RI tensors into the layouts used by the
/// exchange contractions in `matvec`:
/// oo: [naux, occ*occ] -> [occ*naux, occ],
/// vv: [naux, vir*vir] -> [naux*vir, vir],
/// ov: [naux, occ*vir] -> [naux*occ, vir].
pub fn reshape_exchange_tensors(
    oo: &MatrixFull<f64>,
    vv: &MatrixFull<f64>,
    ov: &MatrixFull<f64>,
    occ_size: usize,
    vir_size: usize,
) -> (MatrixFull<f64>, MatrixFull<f64>, MatrixFull<f64>) {
    let num_auxbas = ov.size[0];

    let mut oo_exch = oo.clone();
    oo_exch.reshape([num_auxbas * occ_size, occ_size]);
    let mut oo_exch = oo_exch.transpose_and_drop();
    oo_exch.reshape([occ_size * num_auxbas, occ_size]);

    let mut vv_exch = vv.clone();
    vv_exch.reshape([num_auxbas * vir_size, vir_size]);

    let mut ov_exch = ov.clone();
    ov_exch.reshape([num_auxbas * occ_size, vir_size]);

    (oo_exch, vv_exch, ov_exch)
}

/// HF-exchange coefficients of a range-separated hybrid for the TDDFT response.
///
/// Returns `None` unless the DFA is a range-separated hybrid. Otherwise
/// returns `(omega, coeff_full, coeff_sr)` such that the HF exchange of the
/// response reads `coeff_full*K_full + coeff_sr*K_SR` with
/// `coeff_full = c_LR` and `coeff_sr = c_SR - c_LR`, mirroring the
/// ground-state Fock build in `scf_io`.
pub fn rsh_exchange_coeffs(scf: &SCF) -> Option<(f64, f64, f64)> {
    let (omega, alpha_lr, _beta) = scf.mol.xc_data.rsh_params()?;
    let c_sr = scf.mol.xc_data.dfa_hybrid_scf; // = c_SR = alpha + beta for RSH
    Some((omega, alpha_lr, c_sr - alpha_lr))
}


/// Compute dipole moment integrals in MO basis for TDDFT
///
/// Returns a matrix of shape [3, occ_size * vir_size] containing
/// transition dipole moments mu_{ia}^{d} for each direction d.
///
/// Uses ri_bse::dipoles for integral access (no QP dependency needed
/// for the integral functions themselves).
pub fn compute_tddft_dipole_matrix(
    scf: &SCF,
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    _homo: usize,
    lumo: usize,
) -> MatrixFull<f64> {
    // Get AO dipole integrals (no QP dependency)
    let ao_dip = crate::ri_bse::dipoles::obtain_ao_dips(scf, None);

    let eigvec = &scf.eigenvectors[0];
    let mut dipole_matrix = MatrixFull::new([3, occ_size * vir_size], 0.0);

    for i in 0..occ_size {
        for a in 0..vir_size {
            let mu = crate::ri_bse::dipoles::obtain_mu_ia(
                eigvec, &ao_dip, start_mo + i, lumo + a);
            for d in 0..3 {
                dipole_matrix[[d, i + a * occ_size]] = mu[d];
            }
        }
    }

    dipole_matrix
}

/// Compute MO-basis transition dipoles for unrestricted TDDFT.
///
/// The returned matrix has shape [3, n0+n1], where n0/n1 are the alpha/beta
/// occupied-virtual dimensions.  The column order is the same as the
/// concatenated unrestricted TDDFT vector: all alpha pairs first, then all
/// beta pairs.
pub fn compute_tddft_dipole_matrix_unrestricted(
    scf: &SCF,
    start_mo: [usize; 2],
    occ_size: [usize; 2],
    vir_size: [usize; 2],
    lumo: [usize; 2],
) -> MatrixFull<f64> {
    let ao_dip = crate::ri_bse::dipoles::obtain_ao_dips(scf, None);
    let n0 = occ_size[0] * vir_size[0];
    let n1 = occ_size[1] * vir_size[1];
    let mut dipole_matrix = MatrixFull::new([3, n0 + n1], 0.0);

    for spin in 0..2 {
        let eigvec = &scf.eigenvectors[spin];
        let offset = if spin == 0 { 0 } else { n0 };
        for i in 0..occ_size[spin] {
            for a in 0..vir_size[spin] {
                let idx = offset + i + a * occ_size[spin];
                let mu = crate::ri_bse::dipoles::obtain_mu_ia(
                    eigvec, &ao_dip, start_mo[spin] + i, lumo[spin] + a);
                for d in 0..3 {
                    dipole_matrix[[d, idx]] = mu[d];
                }
            }
        }
    }
    dipole_matrix
}
