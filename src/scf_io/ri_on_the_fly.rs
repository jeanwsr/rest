use rstsr::prelude::*;
use rest_libcint::prelude::*;
use tensors::{BasicMatrix, MatrixFull, MatrixUpper};
use crate::molecule_io::Molecule;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

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
    let partition = crate::grad::rhf::blocksize_partition(&aux_loc, block_size);

    // int2c2e (may be stored in SCF iteration, generate on-the-fly costs some but not that much)
    let tsr_int2c2e = {
        let shls_slice = [[n_basis_shell, n_basis_shell + n_auxbas_shell], [n_basis_shell, n_basis_shell + n_auxbas_shell]];
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
        let shls_slice = [[0, n_basis_shell], [0, n_basis_shell], [n_basis_shell + shl0 as i32, n_basis_shell + shl1 as i32]];
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
        let shls_slice = [[0, n_basis_shell], [0, n_basis_shell], [n_basis_shell + shl0 as i32, n_basis_shell + shl1 as i32]];
        let int3c2e_batch = {
            let (out, shape) = mol.integral_s2ij::<int3c2e>(Some(&shls_slice));
            rt::asarray((out, shape.f(), &device))
        };
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        js_tp.matmul_from(&int3c2e_batch, &scr_js.i(slc), 1.0, 1.0);
        idx_ao += nbatch_ao;
    }
    
    // returns upper triangular part
    js_tp
}

pub(crate) fn generate_vj_ri_direct(dms: &[MatrixFull<f64>], mol_obj: &Molecule, block_size: usize) -> Vec<MatrixUpper<f64>> {
    // dm shape: (nao, nao, nset) in f-contig
    let device = DeviceBLAS::default();
    let nao = dms[0].size()[0];
    let nao_tp = (nao + 1) * nao / 2;
    let nset = dms.len();

    // Vec<MatrixFull> -> Tsr
    let mut dms_rstsr = rt::zeros(([nao, nao, nset].f(), &device));
    for (iset, dm) in dms.iter().enumerate() {
        let dm = rt::asarray((&dm.data, [nao, nao].f(), &device));
        dms_rstsr.i_mut((.., .., iset)).assign(dm);
    }

    let js_rstsr = generate_vj_ri_direct_with_rstsr(dms_rstsr.view(), mol_obj, block_size);

    // println!("js_rstsr: {:12.6?}", js_rstsr);

    // Tsr -> Vec<MatrixUpper>
    let mut js = vec![];
    for iset in 0..nset {
        let j = js_rstsr.i((.., iset)).to_owned().raw().clone();
        js.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, j) });
    }

    js
}
