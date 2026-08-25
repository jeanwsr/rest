use super::prelude_dev::*;
use crate::basis_io::etb::*;

pub fn get_cint_mol(mol_obj: &Molecule) -> CInt {
    mol_obj.initialize_cint(false)
}

pub fn get_cint_aux(mol_obj: &Molecule) -> CInt {
    let etb = if mol_obj.ctrl.even_tempered_basis == true {
        let etb_elem = get_etb_elem(&mol_obj.geom, &mol_obj.ctrl.etb_start_atom_number);
        let etb_basis = etb_gen_for_atom_list(&mol_obj, &mol_obj.ctrl.etb_beta, &etb_elem);
        Some(etb_basis)
    } else {
        None
    };

    let (_, atm, bas, env, _, _, _) = Molecule::collect_auxbas(&mol_obj.ctrl, &mol_obj.geom, etb);
    let mut cint = CInt::new();
    cint.initial_r2c(&atm, atm.len() as _, &bas, bas.len() as _, &env);
    cint.set_cint_type(mol_obj.cint_type);
    cint
}

pub fn retrive_tp_dim(nao_tp: usize) -> Result<usize, String> {
    let nao = ((8 * nao_tp + 1).isqrt() - 1) / 2;
    if nao * (nao + 1) / 2 != nao_tp {
        return Err(format!("nao_tp {} is not a valid triangular packed number", nao_tp));
    }
    Ok(nao)
}

/// Generate the density matrix for current SCF component.
///
/// # Parameters
///
/// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
///
/// # Returns
///
/// - `dm0` : shape `[nao, nao]`. The density matrix for current SCF component.
pub fn get_dm0_restricted(mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
    let [_nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
    assert_eq!(mo_occ.shape(), &[nmo], "mo_occ shape not correct.");

    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    &mocc * occ.i((None, ..)) % &mocc.t()
}

/// Generate the orbital-energy weighted density matrix for current SCF component.
///
/// # Parameters
///
/// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
/// - `mo_energy` : shape `[nmo]`. Molecular orbital energies.
///
/// # Returns
///
/// - `dme0` : shape `[nao, nao]`. The orbital-energy weighted density matrix for current SCF
///   component.
pub fn get_dme0_restricted(mo_coeff: TsrView, mo_occ: TsrView, mo_energy: TsrView) -> Tsr {
    let [_nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
    assert_eq!(mo_occ.shape(), &[nmo], "mo_occ shape not correct.");
    assert_eq!(mo_energy.shape(), &[nmo], "mo_energy shape not correct.");

    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let eocc = mo_energy.bool_select(-1, &occidx);
    &mocc * (occ * eocc).i((None, ..)) % &mocc.t()
}
