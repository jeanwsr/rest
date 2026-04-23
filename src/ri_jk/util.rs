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
