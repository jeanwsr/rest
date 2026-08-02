use rest_libcint::CintType;
use crate::basis_io::{BasCell, BasInfo, Basis4Elem};
use crate::ctrl_io::InputKeywords;
use crate::geom_io::{GeomCell, formated_element_name};

pub fn shell_nao(shells: &[BasCell], cint_type: &CintType) -> usize {
    shells.iter().map(|shell| {
        let ang = shell.angular_momentum[0] as usize;
        let bas_num = match cint_type {
            CintType::Cartesian => (ang + 1) * (ang + 2) / 2,
            CintType::Spheric => ang * 2 + 1,
            CintType::Spinor => panic!("Spinor is not yet implemented"),
        };
        bas_num * shell.coefficients.len()
    }).sum()
}

/// Build fdqc_bas and cint_fdqc from cint_bas and cint_type.
pub fn build_fdqc(
    cint_bas: &[Vec<i32>],
    cint_type: &CintType,
) -> (Vec<BasInfo>, Vec<Vec<usize>>) {
    let mut fdqc_bas: Vec<BasInfo> = vec![];
    let mut cint_fdqc: Vec<Vec<usize>> = vec![];
    let mut bas_start = 0_usize;
    cint_bas.iter().enumerate().for_each(|(bas_index, bas_cell)| {
        let atm_index = bas_cell[0] as usize;
        let ang = bas_cell[1] as usize;
        let num_primitive = bas_cell[2] as usize;
        let num_contracted = bas_cell[3] as usize;
        let tmp_bas_num = match cint_type {
            CintType::Cartesian => (ang + 1) * (ang + 2) / 2,
            CintType::Spheric => ang * 2 + 1,
            CintType::Spinor => panic!("Spinor is not yet implemented"),
        };
        let mut tmp_len = 0_usize;
        (0..num_contracted).for_each(|index0| {
            (0..tmp_bas_num).for_each(|index1| {
                tmp_len += 1;
                fdqc_bas.push(BasInfo {
                    bas_name: super::get_basis_name(ang, cint_type, index1),
                    bas_type: if num_primitive == 1 {
                        String::from("Primitive")
                    } else {
                        String::from("Contracted")
                    },
                    elem_index0: atm_index,
                    cint_index0: bas_index,
                    cint_index1: index0 * tmp_bas_num + index1,
                });
            });
        });
        cint_fdqc.push(vec![bas_start, tmp_len]);
        bas_start += tmp_len;
    });
    (fdqc_bas, cint_fdqc)
}

pub fn read_basis_per_atom(ctrl: &InputKeywords, geom: &GeomCell, cint_type: &CintType) -> Vec<Basis4Elem> {
    let mut basis_total: Vec<Basis4Elem> = vec![];

    for (atm_index, atm_elem) in geom.elem.iter().enumerate() {
        let tmp_path = format!("{}/{}.json", &ctrl.basis_path, &formated_element_name(atm_elem));
        let mut tmp_basis = Basis4Elem::parse_json_from_file(tmp_path, &cint_type).unwrap();
        let num_basis_per_atm = shell_nao(&tmp_basis.electron_shells, &cint_type);
        if atm_index != 0 {
            tmp_basis.global_index.0 = basis_total[atm_index - 1].global_index.0 + basis_total[atm_index - 1].global_index.1;
            tmp_basis.global_index.1 = num_basis_per_atm;
        } else {
            tmp_basis.global_index.0 = 0;
            tmp_basis.global_index.1 = num_basis_per_atm;
        }
        basis_total.push(tmp_basis);
    }

    if geom.ghost_bs_elem.len() > 0 {
        let atm_index_start = geom.elem.len();
        for (local_atm_index, atm_elem) in geom.ghost_bs_elem.iter().enumerate() {
            let atm_index = local_atm_index + atm_index_start;
            let tmp_path = format!("{}/{}.json", &ctrl.basis_path, &formated_element_name(atm_elem));
            let mut tmp_basis = Basis4Elem::parse_json_from_file(tmp_path, &cint_type).unwrap();
            let num_basis_per_atm = shell_nao(&tmp_basis.electron_shells, &cint_type);
            if atm_index != 0 {
                tmp_basis.global_index.0 = basis_total[atm_index - 1].global_index.0 + basis_total[atm_index - 1].global_index.1;
                tmp_basis.global_index.1 = num_basis_per_atm;
            } else {
                tmp_basis.global_index.0 = 0;
                tmp_basis.global_index.1 = num_basis_per_atm;
            }
            basis_total.push(tmp_basis);
        }
    }

    basis_total
}
