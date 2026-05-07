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
    let num_state = scf.mol.num_state;
    let mut homo = 0;
    let mut lumo = num_state;
    for i_spin in 0..scf.mol.spin_channel {
        let i_homo = scf.homo[i_spin];
        let i_lumo = scf.lumo[i_spin];
        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }
    // Auto-detect frozen core: freeze orbitals with very negative eigenvalues
    // (deep core 1s, 2s of heavy atoms) which should not participate in
    // low-energy TDDFT excitations.
    let ks = &scf.eigenvalues[0];
    const FROZEN_CORE_THRESHOLD: f64 = -2.0; // Ha — well below valence orbitals
    let start_mo = (scf.mol.start_mo..=homo)
        .take_while(|&i| ks[i] < FROZEN_CORE_THRESHOLD)
        .last()
        .map(|i| i + 1)
        .unwrap_or(scf.mol.start_mo);
    let occ_size = homo - start_mo + 1;
    // Use all virtual orbitals (no energy cutoff)
    let vir_size = num_state - lumo;
    if start_mo > scf.mol.start_mo {
        println!("  TDDFT frozen core: {:.2} Ha threshold, {} orbitals frozen (MO 0..{})",
            FROZEN_CORE_THRESHOLD, start_mo - scf.mol.start_mo, start_mo);
    }
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
    let range_oo = (start_mo..homo + 1, start_mo..homo + 1);
    let range_vv = (lumo..num_state, lumo..num_state);
    let range_ov = (start_mo..homo + 1, lumo..num_state);

    let ranges = if choice_a == 'O' && choice_b == 'O' {
        range_oo
    } else if choice_a == 'V' && choice_b == 'V' {
        range_vv
    } else if choice_a == 'O' && choice_b == 'V' {
        range_ov
    } else {
        panic!("tddft_get_submatrix: invalid choice {}/{}", choice_a, choice_b);
    };

    let vector = scf.generate_ri3mo_rayon_for_multiple_times(ranges.0, ranges.1);
    let matrix: MatrixFull<f64> = vector[0].0.rifull_to_matfull_i_jk();
    println!(
        "TDDFT RI Tensor: {}-{}, Size={:?}, naux={}",
        choice_a, choice_b, matrix.size, matrix.size[0]
    );
    matrix
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
