use log::warn;
use rstsr::prelude::*;
use tensors::matrix::MatrixFull;
use tensors::BasicMatrix;
use crate::basis_io::Basis4Elem;
use crate::constants::{ATM_ENV, ATM_NUC};
use crate::fileop::chkfile;
use crate::molecule_io::Molecule;
use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI};

/// Solve S22 * X = B using Cholesky + triangular solve, with fallback to general solve.
/// Equivalent to pyscf's lib.cho_solve(a, b, strict_sym_pos=False).
fn cho_solve(s22: &MatrixFull<f64>, b: &MatrixFull<f64>, device: &DeviceBLAS) -> MatrixFull<f64> {
    let s22_tsr = s22.to_rstsr_view(device);
    let b_tsr = b.to_rstsr_view(device);

    match rt::linalg::cholesky_f((&s22_tsr, Lower)) {
        Ok(l) => {
            // println!("Cholesky decomposition succeeded, using triangular solve.");
            let y = rt::linalg::solve_triangular((&l, &b_tsr, Lower));
            let x = rt::linalg::solve_triangular((l.t(), &y, Upper));
            let shape: [usize; 2] = x.shape().to_vec().try_into().unwrap();
            MatrixFull::from_vec(shape, x.into_shape(-1).into_vec()).unwrap()
        },
        Err(_) => {
            warn!("Cholesky decomposition failed, falling back to general solve.");
            let x = rt::linalg::solve_general((&s22_tsr, &b_tsr));
            let shape: [usize; 2] = x.shape().to_vec().try_into().unwrap();
            MatrixFull::from_vec(shape, x.into_shape(-1).into_vec()).unwrap()
        }
    }
}

/// Normalize MO columns so that mo^T S mo = I.
/// Equivalent to pyscf:
///   norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
///   mo /= numpy.sqrt(norm)
fn normalize_mo(mo: &mut MatrixFull<f64>, s: &MatrixFull<f64>, device: &DeviceBLAS) {
    let s_tsr = s.to_rstsr_view(device);
    let mo_tsr = mo.to_rstsr_view(device);

    let s_dot_mo = &s_tsr % &mo_tsr;
    let norm = (&mo_tsr * &s_dot_mo).sum_axes(0);

    let norm_vec: Vec<f64> = norm.to_vec();
    let inv_sqrt_norm: Vec<f64> = norm_vec.iter().map(|x| 1.0 / x.sqrt()).collect();
    let inv_sqrt_tsr = rt::asarray((inv_sqrt_norm.clone(), device));

    // println!("mo shape: {:?}, norm shape: {:?}, inv_sqrt_norm shape: {:?}", mo_tsr.shape(), norm.shape(), inv_sqrt_tsr.shape());
    // println!("inv_sqrt_norm: {:?}", inv_sqrt_norm[..5].to_vec());
    let mo_normalized = &mo_tsr * inv_sqrt_tsr.slice((None, ..));

    let shape: [usize; 2] = mo_normalized.shape().to_vec().try_into().unwrap();
    *mo = MatrixFull::from_vec(shape, mo_normalized.into_shape(-1).into_vec()).unwrap();
}

/// Project MO coefficients from a source molecule's basis to a target molecule's basis.
///
/// # Arguments
/// - `mol_target`: the target molecule (basis set 2)
/// - `mol_source`: the source molecule (basis set 1)
/// - `mo_source`: MO coefficients in the source basis
///
/// # Returns
/// - MO coefficients projected to the target basis (C2), normalized so that C2^T S22 C2 = I
pub fn proj_mo(mol_target: &Molecule, mol_source: &Molecule, mo_source: [MatrixFull<f64>;2]) -> [MatrixFull<f64>;2] {
    // S22 = target self-overlap
    let s22_full = mol_target.int_ij_matrixupper("ovlp".to_string()).to_matrixfull().unwrap();

    // S21 = <AO_target|AO_source>
    let s21 = mol_target.int_cross(mol_source, "ovlp".to_string());

    let device = DeviceBLAS::default();
    let s21_tsr = s21.to_rstsr(&device);

    let mut mo_target: [MatrixFull<f64>; 2] = [MatrixFull::empty(), MatrixFull::empty()];
    for spin in 0..2 {
        if mo_source[spin].size()[0] == 0 {
            continue;
        }
        let mo_source_tsr = mo_source[spin].to_rstsr_view(&device);
        let temp_tsr = &s21_tsr % &mo_source_tsr;
        let temp = MatrixFull::from_vec(
            temp_tsr.shape().to_vec().try_into().unwrap(),
            temp_tsr.into_shape(-1).into_vec(),
        ).unwrap();

        mo_target[spin] = cho_solve(&s22_full, &temp, &device);
        normalize_mo(&mut mo_target[spin], &s22_full, &device);
    }
    // mo_target[0].formated_output(5, "full");

    mo_target
}

pub enum GuessAction {
    DirectReuse,
    Project(Molecule),
    Refuse(String),
}

const GEOM_TOL: f64 = 1e-6;

fn ecp_electrons_from_basis(basis4elem: &Option<Vec<Basis4Elem>>) -> usize {
    match basis4elem {
        Some(v) => v.iter().fold(0, |acc, i| acc + i.ecp_electrons.unwrap_or(0)),
        None => 0,
    }
}

pub fn decide_guess(chkfile: &String, mol_target: &Molecule) -> GuessAction {
    let (loaded_nbasis, loaded_nmo, loaded_spin_channel, loaded_spin, loaded_charge) =
        chkfile::load_basic(chkfile).unwrap();

    let (cint_raw_data, ecp_raw, basis4elem, cint_type, _, _) = chkfile::reconstruct_cint_data(chkfile, None);
    let (source_atm, source_bas, source_env) = cint_raw_data.unwrap();

    let source_geom = chkfile::load_geom(chkfile);

    let source_natm = source_geom.as_ref()
        .map(|g| g.elem.len())
        .unwrap_or(source_atm.len());
    let target_natm = mol_target.geom.elem.len();
    if source_natm != target_natm {
        return GuessAction::Refuse(format!(
            "atom count mismatch: chkfile has {}, target has {}",
            source_natm, target_natm
        ));
    }

    if let Some(ref sgeom) = source_geom {
        for i in 0..source_natm {
            if sgeom.elem[i] != mol_target.geom.elem[i] {
                return GuessAction::Refuse(format!(
                    "element mismatch at atom {}: chkfile element={}, target element={}",
                    i, sgeom.elem[i], mol_target.geom.elem[i]
                ));
            }
        }
    } else {
        for i in 0..source_natm {
            let s_z = source_atm[i][ATM_NUC] as f64;
            let t_z = mol_target.cint_atm[i][ATM_NUC] as f64;
            if (s_z - t_z).abs() > 1e-12 {
                return GuessAction::Refuse(format!(
                    "element mismatch at atom {}: chkfile Z={}, target Z={}",
                    i, s_z, t_z
                ));
            }
        }
    }

    let s_cint_type = cint_type.unwrap_or(mol_target.cint_type);
    if mol_target.cint_type != s_cint_type {
        return GuessAction::Refuse(format!(
            "CintType mismatch: target {:?}, source {:?}",
            mol_target.cint_type, s_cint_type
        ));
    }

    let source_ecp_electrons = ecp_electrons_from_basis(&basis4elem);
    if mol_target.ecp_electrons != source_ecp_electrons {
        return GuessAction::Refuse(format!(
            "ECP electron count mismatch: target {}, source {}",
            mol_target.ecp_electrons, source_ecp_electrons
        ));
    }

    let mut geom_diff = true;
    if let Some(ref sgeom) = source_geom {
        if sgeom.unit == mol_target.geom.unit
            && sgeom.position.size() == mol_target.geom.position.size()
        {
            geom_diff = sgeom.position.iter()
                .zip(mol_target.geom.position.iter())
                .any(|(s, t)| (s - t).abs() > GEOM_TOL);
        }
    } else {
        geom_diff = false;
        for i in 0..source_atm.len() {
            let s_ptr = source_atm[i][ATM_ENV] as usize;
            let t_ptr = mol_target.cint_atm[i][ATM_ENV] as usize;
            for d in 0..3 {
                if (source_env[s_ptr + d] - mol_target.cint_env[t_ptr + d]).abs() > GEOM_TOL {
                    geom_diff = true;
                    break;
                }
            }
            if geom_diff { break; }
        }
    }

    let mut mol_source = Molecule::init_mol();
    mol_source.cint_type = cint_type.unwrap_or(mol_target.cint_type);
    mol_source.ctrl.spin = loaded_spin.unwrap_or(mol_target.ctrl.spin);
    mol_source.ctrl.charge = loaded_charge.unwrap_or(mol_target.ctrl.charge);
    mol_source.ctrl.print_level = mol_target.ctrl.print_level;
    mol_source.cint_atm = source_atm;
    mol_source.cint_bas = source_bas;
    mol_source.cint_env = source_env;
    mol_source.cint_ecpbas = ecp_raw;
    mol_source.num_state = loaded_nmo;
    mol_source.num_basis = loaded_nbasis;
    mol_source.spin_channel = loaded_spin_channel;

    let basis_diff = loaded_nbasis != mol_target.num_basis || loaded_nmo != mol_target.num_state;

    if basis_diff || geom_diff {
        return GuessAction::Project(mol_source);
    }

    let s22 = mol_target.int_ij_matrixupper("ovlp".to_string()).to_matrixfull().unwrap();
    let s21 = mol_target.int_cross(&mol_source, "ovlp".to_string());
    let s22_norm2: f64 = s22.iter().map(|a| a.powi(2)).sum();
    let diff_norm2: f64 = s22.iter().zip(s21.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let rel = (diff_norm2 / s22_norm2).sqrt();
    if rel > 1e-5 {
        return GuessAction::Project(mol_source);
    }
    GuessAction::DirectReuse
}