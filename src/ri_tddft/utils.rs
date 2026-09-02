/// TDDFT utility functions (no dependency on quasiparticle_methods)
///
/// Provides self-contained replacements for functions from ri_gw and ri_bse
/// that require QuasiParticle control parameters.

use rest_tensors::{MatrixFull, RIFull};
use crate::scf_io::{SCF, SCFType};

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

/// Per-spin TDDFT orbital parameters for an unrestricted (UKS/UHF) reference.
///
/// Each spin channel carries its own occupied/virtual window; the TDDFT
/// amplitude space is the concatenation `[z_alpha (dim_a); z_beta (dim_b)]`
/// with `dim_s = occ_size_s * vir_size_s` (PySCF `tdscf/uhf.py` layout).
#[derive(Clone, Copy, Debug)]
pub struct TddftSector {
    pub start_mo: usize,
    pub num_state: usize,
    pub occ_size: usize,
    pub vir_size: usize,
    pub homo: usize,
    pub lumo: usize,
}

impl TddftSector {
    /// Number of single-spin amplitudes (occ*vir) of this sector.
    pub fn dim(&self) -> usize {
        self.occ_size * self.vir_size
    }
}

/// Sector parameter list for either reference type: one entry for RHF (the
/// restricted window), two for UHF. This is the single entry point kernel code
/// should use — sector loops over `TddftSector` then handle both references
/// uniformly.
pub fn tddft_sector_params(scf: &SCF) -> Vec<TddftSector> {
    if scf.scftype == SCFType::UHF {
        tddft_occupation_parameters_u(scf).to_vec()
    } else {
        let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
            tddft_occupation_parameters(scf);
        vec![TddftSector { start_mo, num_state, occ_size, vir_size, homo, lumo }]
    }
}

/// Per-spin TDDFT orbital parameters (unrestricted reference).
///
/// Mirrors [`tddft_occupation_parameters`] but resolves frozen-core and
/// virtual-cutoff windows independently on each spin channel's eigenvalues,
/// instead of mixing `homo = max` / `lumo = min` across spins.
/// Returns `[alpha_sector, beta_sector]`; a sector with no occupied orbitals
/// (e.g. an empty beta channel) yields `occ_size == 0` (`dim() == 0`).
pub fn tddft_occupation_parameters_u(scf: &SCF) -> [TddftSector; 2] {
    let cutoff = scf.mol.ctrl.tddft.as_ref()
        .map(|c| c.tddft_cutoff_energy)
        .unwrap_or(1.0e6);
    let mut sectors = [TddftSector {
        start_mo: 0, num_state: 0, occ_size: 0, vir_size: 0, homo: 0, lumo: 0,
    }; 2];
    for i_spin in 0..2 {
        let ks = &scf.eigenvalues[i_spin];
        let homo = scf.homo[i_spin];
        let lumo = scf.lumo[i_spin];
        let mut num_state = scf.mol.num_state;
        // Auto-detect frozen core (same heuristic/threshold as the restricted path).
        const FROZEN_CORE_THRESHOLD: f64 = -2.0; // Ha
        let start_mo = (scf.mol.start_mo..=homo)
            .take_while(|&i| ks[i] < FROZEN_CORE_THRESHOLD)
            .last()
            .map(|i| i + 1)
            .unwrap_or(scf.mol.start_mo);
        // Guard degenerate channels: an EMPTY spin channel carries the
        // homo=0 sentinel (scf_io/occupation.rs), indistinguishable from a
        // real 1-electron HOMO — disambiguate via the electron count.
        // num_elec = [total, alpha, beta].
        let nocc_s = scf.mol.num_elec[i_spin + 1];
        let occ_size = if nocc_s >= 0.5 && homo >= start_mo { homo - start_mo + 1 } else { 0 };
        if cutoff < 1.0e5 {
            num_state = ks.iter().filter(|x| **x < cutoff).count();
            if num_state < homo + 1 {
                num_state = homo + 1;
            }
        }
        let lumo_eff = lumo.max(start_mo + occ_size).min(num_state);
        let vir_size = num_state.saturating_sub(lumo_eff);
        sectors[i_spin] = TddftSector {
            start_mo, num_state, occ_size, vir_size, homo, lumo: lumo_eff,
        };
    }
    sectors
}

/// Compute spin-resolved transition dipole integrals in the concatenated
/// (unrestricted) MO amplitude basis.
///
/// Returns `[3, dim_a + dim_b]` with the per-sector blocks laid out as
/// `mu_ia^{sigma}` at column `base_s + i + a*occ_size_s`, matching the
/// concatenated amplitude ordering.
pub fn compute_tddft_dipole_matrix_u(
    scf: &SCF,
    sectors: &[TddftSector; 2],
) -> MatrixFull<f64> {
    let ao_dip = crate::ri_bse::dipoles::obtain_ao_dips(scf, None);
    let dim_u = sectors[0].dim() + sectors[1].dim();
    let mut dipole_matrix = MatrixFull::new([3, dim_u], 0.0);
    let mut base = 0usize;
    for i_spin in 0..2 {
        let sec = &sectors[i_spin];
        let eigvec = &scf.eigenvectors[i_spin];
        for a in 0..sec.vir_size {
            for i in 0..sec.occ_size {
                let mu = crate::ri_bse::dipoles::obtain_mu_ia(
                    eigvec, &ao_dip, sec.start_mo + i, sec.lumo + a);
                for d in 0..3 {
                    dipole_matrix[[d, base + i + a * sec.occ_size]] = mu[d];
                }
            }
        }
        base += sec.dim();
    }
    dipole_matrix
}

/// Normalize unrestricted TDDFT eigenpairs (PySCF `tdscf/uhf.py` convention):
/// divide by `sqrt(|X|^2 - |Y|^2)` for full LR (no restricted `1/sqrt(2)` and
/// no RKS-specific factors); plain `1/|x|` for TDA.
pub fn normalize_u(vector: &[f64], tda: bool) -> Vec<f64> {
    let norm2 = if tda {
        vector.iter().map(|x| x * x).sum::<f64>()
    } else {
        let n = vector.len() / 2;
        let xn: f64 = vector[..n].iter().map(|x| x * x).sum();
        let yn: f64 = vector[n..].iter().map(|x| x * x).sum();
        xn - yn
    };
    let norm = norm2.abs().sqrt().max(1.0e-12);
    vector.iter().map(|x| x / norm).collect()
}

/// Transition dipole square for unrestricted TDDFT (PySCF convention): the
/// spin-summed transition dipole is `mu_d = sum_{ia} (X+Y)_{ia} mu^d_{ia}`
/// summed over BOTH spin sectors — no restricted closed-shell factor of 2.
/// `dipole_matrix` is `[3, dim_u]` (see [`compute_tddft_dipole_matrix_u`]);
/// `vector` is the eigenpair `[X; Y]` (TDA: just `X`).
pub fn transition_dipole_square_u(
    dipole_matrix: &MatrixFull<f64>,
    vector: &[f64],
    tda: bool,
) -> f64 {
    let size = dipole_matrix.size[1];
    let v: Vec<f64> = if tda {
        vector.to_vec()
    } else {
        let x = &vector[..size];
        let y = &vector[size..];
        x.iter().zip(y.iter()).map(|(a, b)| a + b).collect()
    };
    let mut mu = [0.0f64; 3];
    for d in 0..3 {
        mu[d] = (0..size).map(|idx| dipole_matrix[[d, idx]] * v[idx]).sum();
    }
    println!("\tDipole Moment Components:\n\tx:{}, y:{}, z:{}", mu[0], mu[1], mu[2]);
    mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]
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
