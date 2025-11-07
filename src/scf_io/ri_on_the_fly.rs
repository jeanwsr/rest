use crate::molecule_io::Molecule;
use crate::utilities::memory_batch::blocksize_partition;
use crate::utilities::rstsr_util::*;
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use tensors::{MatrixFull, MatrixUpper};

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

/// Generate Coulomb (J) matrix on-the-fly using RI direct method.
///
/// This function is low-level implementation, using RSTSR tensors as input and output. For
/// high-level interface (using rest_tensors as input and output), please refer to
/// [`generate_vj_ri_direct`].
///
/// # Parameters
///
/// - `dms`: [`TsrView<f64>`]
///
///   - Density matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///   - This matrix is assumed to be in AO basis.
///   - This matrix is assumed to be symmetric. We will not perform symmetry check or symmetrize
///     operation.
///
/// - `mol_obj`: [`Molecule`]
///
///   - Please make sure auxiliary basis is assigned to this molecule object.
///
/// - `block_size`: `usize`
///
///   - Block size for auxiliary basis partitioning. This value controls memory usage.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Coulomb (J) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - J matrices are symmetric by definition in real arithmetic.
///
/// # Formula and Algorithm
///
/// $$
/// \begin{align*}
/// \mathscr{T}_P^\text{1} [\mathbf{D}^\mathbb{A}] &= \sum_{\kappa \lambda} g_{\kappa \lambda, P}
/// D_{\kappa \lambda}^\mathbb{A} \tag{eq.1} \\
/// \mathscr{T}_P^\text{2} [\mathbf{D}^\mathbb{A}] &= \sum_{Q} (\mathbf{J}^{-1})_{PQ}
/// \mathscr{T}_Q^\text{1} [\mathbf{D}^\mathbb{A}] \tag{eq.2} \\
/// J_{\mu \nu} [\mathbf{D}^\mathbb{A}] &= \sum_{P} g_{\mu \nu, P} \mathscr{T}_P^\text{2}
/// [\mathbf{D}^\mathbb{A}] \tag{eq.3}
/// \end{align*}
/// $$
///
/// - AO indices ($\mu \nu$, $\kappa, \lambda$) are in packed upper-triangular format.
/// - Auxiliary basis are batched, sparately in (eq.1) and (eq.3).
/// - (eq.2) is solved using general linear solver.
///
/// Fixed memory requirement:
///
/// - Storage of $\mathscr{T}_P^\text{1} [\mathbf{D}^\mathbb{A}]$ and $\mathscr{T}_P^\text{2}
///   [\mathbf{D}^\mathbb{A}]$, which costs (naux * nset * 2).
/// - Storage of decomposed or inversed 2c-2e ERI $J_{PQ}$, which costs approximately (naux * naux).
///
/// Batched memory requirement (controlled by `block_size`):
///
/// - Storage of 3c-2e ERI $g_{\kappa \lambda, P}$ for a batch of auxiliary basis, which costs
///   (nao_tp * block_size).
pub(crate) fn generate_vj_ri_direct_with_rstsr(dms: TsrView<f64>, mol_obj: &Molecule, block_size: usize) -> Tsr<f64> {
    // dm shape: (nao, nao, nset) in f-contig
    assert!(dms.ndim() == 3, "DM must have 3 dimensions");

    let mut mol = mol_obj.initialize_cint(true);
    let mut aux = mol_obj.make_auxmol_fake().initialize_cint(false);

    let nset = dms.shape()[2];
    let nao = dms.shape()[0];
    let nao_tp = (nao + 1) * nao / 2;
    let naux = aux.cgto_loc().last().unwrap().clone();
    let device = dms.device().clone();
    let n_basis_shell = mol_obj.cint_bas.len() as i32;
    let n_auxbas_shell = mol_obj.cint_aux_bas.len() as i32;

    // get partition
    let aux_loc = &mol.cgto_loc()[(n_basis_shell as usize)..];
    let partition = blocksize_partition(&aux_loc, block_size);

    // int2c2e (may be stored in SCF iteration, generate on-the-fly costs some but not that much)
    let tsr_int2c2e = {
        let shls_slice =
            [[n_basis_shell, n_basis_shell + n_auxbas_shell], [n_basis_shell, n_basis_shell + n_auxbas_shell]];
        let (out, shape) = mol.integral_s1::<int2c2e>(Some(&shls_slice));
        rt::asarray((out, shape.f(), &device))
    };

    // pack density matrix
    let mut dms_tp = rt::zeros(([nao_tp, nset].f(), &device));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm = 2.0_f64 * &dm;
        dm.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // generate scr_j
    let mut scr_j = rt::zeros(([naux, nset].f(), &device));
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        let nbatch_ao = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice =
            [[0, n_basis_shell], [0, n_basis_shell], [n_basis_shell + shl0 as i32, n_basis_shell + shl1 as i32]];
        let int3c2e_batch = {
            let (out, shape) = mol.integral_s2ij::<int3c2e>(Some(&shls_slice));
            rt::asarray((out, shape.f(), &device))
        };
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        scr_j.i_mut(slc).matmul_from(&int3c2e_batch.t(), &dms_tp, 1.0, 0.0);
        idx_ao += nbatch_ao;
    }

    // solve scr_j
    let scr_js = rt::linalg::solve_general((tsr_int2c2e, scr_j));

    // generate j contribution
    let mut js_tp = rt::zeros(([nao_tp, nset].f(), &device));
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        let nbatch_ao = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice =
            [[0, n_basis_shell], [0, n_basis_shell], [n_basis_shell + shl0 as i32, n_basis_shell + shl1 as i32]];
        let int3c2e_batch = {
            let (out, shape) = mol.integral_s2ij::<int3c2e>(Some(&shls_slice));
            rt::asarray((out, shape.f(), &device))
        };
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        js_tp.matmul_from(&int3c2e_batch, &scr_js.i(slc), 1.0, 1.0);
        idx_ao += nbatch_ao;
    }

    // returns symmetrized part
    let mut js = rt::zeros(([nao, nao, nset].f(), &device));
    for iset in 0..nset {
        js.i_mut((.., .., iset)).assign(js_tp.i((.., iset)).unpack_triu(FlagSymm::Sy));
    }
    js
}

pub(crate) fn generate_vj_ri_direct(
    dms: &[MatrixFull<f64>],
    mol_obj: &Molecule,
    block_size: usize,
) -> Vec<MatrixUpper<f64>> {
    // dm shape: (nao, nao, nset) in f-contig
    let device = DeviceBLAS::default();
    let dms_rstsr = dms.to_rstsr(&device);

    let nao = dms_rstsr.shape()[0];
    let nset = dms_rstsr.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    let js_rstsr = generate_vj_ri_direct_with_rstsr(dms_rstsr.view(), mol_obj, block_size);

    // Tsr -> Vec<MatrixUpper>
    let mut js = vec![];
    for iset in 0..nset {
        let j = js_rstsr.i((.., iset)).pack_tri(Upper).into_vec();
        js.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, j) });
    }

    js
}

#[cfg(test)]
mod debug {
    use super::*;
    use crate::scf_io::SCF;

    #[test]
    fn test_nh3() {
        let scf_data = initialize_nh3();
        let device = DeviceBLAS::default();

        let dm = [&scf_data.density_matrix[0]].as_ref().to_rstsr(&device);
        let mol = &scf_data.mol;
        // full batch
        let j_rstsr = generate_vj_ri_direct_with_rstsr(dm.view(), mol, 10000);
        let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
        let ref_fp = 37.83424292927407;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // small batch
        let j_rstsr = generate_vj_ri_direct_with_rstsr(dm.view(), mol, 16);
        let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
        let ref_fp = 37.83424292927407;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);
    }

    fn initialize_nh3() -> SCF {
        let input_token = r##"
[ctrl]
     print_level =          2
     method =               "hf"
     basis_path =           "basis-set-pool/def2-TZVP"
     auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
     eri_type =             "ri-v"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess=         "sad"
     mixer =                "diis"
     num_threads =          16

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  0.0  1.5  1.0
        H  1.4  1.1  0.0
        H  1.2  0.0  1.3
    """
"##;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (mut ctrl, mut geom) = crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol = Molecule::build_native(ctrl, geom, None).unwrap();
        let mut scf_data = crate::scf_io::SCF::build(mol, &None);
        crate::scf_io::scf_without_build(&mut scf_data, &None);
        return scf_data;
    }
}
