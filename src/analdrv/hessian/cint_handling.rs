use crate::analdrv::prelude::*;
use rest_libcint::util::ShlsSlice;

/// A wrapper around [`CInt::integrate`] that directly transform the output and shape to tensor.
///
/// This only handles single molecule integrals. For cross integrals, see [`hess_intor_cross`].
///
/// # Notes
///
/// The following notes also apply to [`hess_intor_cross`].
///
/// Note that this wrapper **only works in hessian-related tasks**. We will transform integrators
/// like `int2c2e_ipip1` original shape `[naux, naux, 9]` to [naux, naux, 3, 3]` to make it more
/// intuitive to use. For other integrators, the shape will be unchanged.
///
/// Also note that, for second order derivative, for example of `int3c2e_ip1ip2` $(\partial_t \mu
/// \nu | \partial_s P)$, the returned shape is `[nao, nao, naux, 3, 3]`, denoting the indices of
/// $(\mu, \nu, P, s, t)$. Please be very careful about the last two dimensions, which are of
/// indices `[s, t]` for column major.
pub fn hess_intor(
    mol: &CInt,
    intor_name: &str,
    symm: &str,
    shls_slice: impl Into<ShlsSlice>,
    device: &DeviceBLAS,
) -> Tsr {
    let (out, shape) = mol.integrate(intor_name, symm, shls_slice.into()).into();
    let ip_matches = intor_name.matches("ip").count();
    let shape = match ip_matches {
        0 | 1 => shape,
        2 => {
            // check last dimension is 9
            assert_eq!(shape.last(), Some(&9), "For integrator with 2 'ip' in name, the last dimension should be 9.");
            // transform last dimension from 9 to (3, 3)
            let mut new_shape = shape.clone();
            new_shape.pop();
            new_shape.push(3);
            new_shape.push(3);
            new_shape
        },
        _ => panic!("Unsupported integrator with more than 2 'ip' in name."),
    };
    rt::asarray((out, shape, device))
}

/// A wrapper around [`CInt::integrate_cross`] that directly transform the output and shape to
/// tensor.
///
/// Notes see also [`hess_intor`].
pub fn hess_intor_cross(
    mol_list: &[&CInt],
    intor_name: &str,
    symm: &str,
    shls_slice: impl Into<ShlsSlice>,
    device: &DeviceBLAS,
) -> Tsr {
    let (out, shape) = CInt::integrate_cross(intor_name, mol_list, symm, shls_slice.into()).into();
    let ip_matches = intor_name.matches("ip").count();
    let shape = match ip_matches {
        0 | 1 => shape,
        2 => {
            // check last dimension is 9
            assert_eq!(shape.last(), Some(&9), "For integrator with 2 'ip' in name, the last dimension should be 9.");
            // transform last dimension from 9 to (3, 3)
            let mut new_shape = shape.clone();
            new_shape.pop();
            new_shape.push(3);
            new_shape.push(3);
            new_shape
        },
        _ => panic!("Unsupported integrator with more than 2 'ip' in name."),
    };
    rt::asarray((out, shape, device))
}

/// A wrapper that generates 3c-2e ERIs.
///
/// This returns a closure that takes `shls_aux` as input. So, the batch is always on the auxiliary
/// shell dimension. Also note input is shell index, not AO index.
pub fn generator_hess_intor_j3c_by_aux<'a>(
    mol: &'a CInt,
    aux: &'a CInt,
    intor_name: &'a str,
    symm: &'a str,
    device: &DeviceBLAS,
) -> impl Fn([usize; 2]) -> Tsr + 'a {
    let shls_mol = [0, mol.nbas()];
    let device = device.clone();
    move |shls_aux: [usize; 2]| {
        hess_intor_cross(&[mol, mol, aux], intor_name, symm, [shls_mol, shls_mol, shls_aux], &device)
    }
}

pub fn get_ecp_atoms(mol: &CInt) -> Vec<usize> {
    const ATOM_OF: usize = rest_libcint::ffi::cint_ffi::ATOM_OF as usize;
    // remove duplicates and sort
    mol.ecpbas.iter().map(|&ecpbas| ecpbas[ATOM_OF] as usize).sorted().dedup().collect_vec()
}

/// Filter the atom-slice array according to an optional list of atom indices.
///
/// When `atm_list` is `None`, returns the full slices (length `mol.natm()`) and all indices.
/// When `atm_list` is `Some(&[i, j, ...])`, returns slices for only those atoms (length
/// `list.len()`) and the same list as the index mapping (local → global).
pub fn filter_aoslices(mol: &CInt, atm_list: Option<&[usize]>) -> (Vec<[usize; 4]>, Vec<usize>) {
    let full_slices = mol.aoslice_by_atom();
    match atm_list {
        None => (full_slices, (0..mol.natm()).collect()),
        Some(list) => {
            let slices = list.iter().map(|&i| full_slices[i]).collect();
            let indices = list.to_vec();
            (slices, indices)
        },
    }
}
