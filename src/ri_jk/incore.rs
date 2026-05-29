use super::prelude_dev::*;

/// Estimate memory requirement for VK RI incore method with MO coefficients and occupations.
///
/// This function corresponds to [`generate_vk_ri_incore_coeff_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `nocc_max`: Maximum number of occupied molecular orbitals among all sets.
/// - `nset`: Number of density matrix sets.
pub fn mem_estimate_vk_ri_incore_coeff(nao: usize, nocc_max: usize, nset: usize) -> MemEstimate {
    // int3c2e batch
    let batched = nao * nocc_max;
    // ks, occ_coeff_list
    let fixed = nao * nao * nset + nao * nocc_max * nset;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}

pub fn mem_estimate_vk_ri_incore_dm(nao: usize) -> MemEstimate {
    // cderi_half, cderi_batch
    let batched = nao * nao * 2;
    // ks
    let fixed = nao * nao;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}
