use super::prelude_dev::*;
use super::pure_direct::*;

/// Estimate memory requirement for VJ RI direct method.
///
/// This function corresponds to [`generate_vj_ri_direct_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `naux`: Number of auxiliary basis functions.
/// - `nset`: Number of density matrix sets.
pub const fn mem_estimate_vj_ri_direct(nao: usize, naux: usize, nset: usize) -> MemEstimate {
    let nao_tp = (nao + 1) * nao / 2;
    // int3c2e batch
    let batched = nao_tp;
    // scr_j, int2c2e, js_tp / dms_tp, js
    let fixed = naux * nset + naux * naux + nao_tp * nset + nao * nao * nset;
    let thread = 0;
    MemEstimate { batched, fixed, thread }
}

pub fn generate_vj_ri_direct(dms: &[MatrixFull<f64>], mol_obj: &Molecule, batch_size: usize) -> Vec<MatrixUpper<f64>> {
    // dm shape: (nao, nao, nset) in f-contig
    let device = DeviceBLAS::default();
    let dms_rstsr = dms.to_rstsr(&device);

    let nao = dms_rstsr.shape()[0];
    let nset = dms_rstsr.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    let mol = util::get_cint_mol(mol_obj);
    let aux = util::get_cint_aux(mol_obj);

    let js_rstsr = get_vj_ri_direct(dms_rstsr.view(), &mol, &aux, batch_size);

    // Tsr -> Vec<MatrixUpper>
    let mut js = vec![];
    for iset in 0..nset {
        let j = js_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        js.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, j) });
    }
    js
}

pub fn generate_vk_ri_semi_direct_coeff(
    scaling_factor: f64,
    mo_coeff: &[MatrixFull<f64>],
    mo_occ: &[Vec<f64>],
    mol_obj: &Molecule,
    omega: Option<f64>,
    batch_size: usize,
) -> Vec<MatrixUpper<f64>> {
    // mo_coeff shape: (nao, nmo, nset) in f-contig
    let device = DeviceBLAS::default();
    let mo_coeff_rstsr = mo_coeff.to_rstsr(&device);
    let mo_occ_rstsr = mo_occ.to_rstsr(&device);

    let nao = mo_coeff_rstsr.shape()[0];
    let nset = mo_coeff_rstsr.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    let mut mol = util::get_cint_mol(mol_obj);
    let mut aux = util::get_cint_aux(mol_obj);

    if let Some(omega) = omega {
        mol.set_omega(omega);
        aux.set_omega(omega);
    }

    let mut ks_rstsr = get_vk_ri_semi_direct_coeff(mo_coeff_rstsr.view(), mo_occ_rstsr.view(), &mol, &aux, batch_size);
    if (scaling_factor - 1.0).abs() > f64::EPSILON {
        ks_rstsr *= scaling_factor;
    }

    // Tsr -> Vec<MatrixUpper>
    let mut ks = vec![];
    for iset in 0..nset {
        let k = ks_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        ks.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, k) })
    }
    ks
}

pub fn generate_vk_ri_direct_dm(
    scaling_factor: f64,
    dms: &[MatrixFull<f64>],
    mol_obj: &Molecule,
    omega: Option<f64>,
    batch_size: usize,
) -> Vec<MatrixUpper<f64>> {
    let device = DeviceBLAS::default();
    let dms = dms.to_rstsr(&device);

    let nao = dms.shape()[0];
    let nset = dms.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    let mut mol = util::get_cint_mol(mol_obj);
    let mut aux = util::get_cint_aux(mol_obj);

    if let Some(omega) = omega {
        mol.set_omega(omega);
        aux.set_omega(omega);
    }

    // shape sanity check
    assert_eq!(dms.shape(), &[nao, nao, nset], "Density matrices must have shape (nao, nao, nset)");

    let mut ks_rstsr = get_vk_ri_direct_dm(dms.view(), &mol, &aux, batch_size);
    if (scaling_factor - 1.0).abs() > f64::EPSILON {
        ks_rstsr *= scaling_factor;
    }

    // Tsr -> Vec<MatrixUpper>
    let mut ks = vec![];
    for iset in 0..nset {
        let k = ks_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        ks.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, k) })
    }
    ks
}
