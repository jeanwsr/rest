#![warn(unused)]
use super::prelude_dev::*;
use super::*;
use crate::scf_io::SCF;
use crate::utilities::TimeRecords;

use rt::blas::{BlasFloat, LapackDriverAPI};

type Tsr<T> = Tensor<T, DeviceBLAS>;

pub fn obtain_cderi_xvo_restricted<T>(
    scf_data: &SCF,
    timerecords: &mut TimeRecords,
    eigenvectors: Option<&MatrixFull<f64>>,
    row_range: Option<&[usize]>,
    col_range: Option<&[usize]>,
) -> Tsr<T>
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = DeviceBLAS::default();

    let mo_coeff = match eigenvectors {
        None => (&scf_data.eigenvectors[0]).to_rstsr(&device),
        Some(eigenvectors) => eigenvectors.to_rstsr(&device),
    };

    // apply occ and vir orbital indices
    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo[0];
    let nmo = mo_coeff.shape()[1];
    let occ_coeff = match col_range {
        None => {
            let occ_list: Vec<usize> = (idx_core..idx_lumo).collect();
            mo_coeff.index_select(-1, &occ_list)
        },
        Some(indices) => mo_coeff.index_select(-1, indices),
    };
    let vir_coeff = match row_range {
        None => {
            let vir_list: Vec<usize> = (idx_lumo..nmo).collect();
            mo_coeff.index_select(-1, &vir_list)
        },
        Some(indices) => mo_coeff.index_select(-1, indices),
    };

    let nocc = occ_coeff.shape()[1];
    let nvir = vir_coeff.shape()[1];
    let nao = mo_coeff.shape()[0];
    let ns = nocc.max(nvir);
    let nthreads = rayon::current_num_threads();

    if let Some(rimatr) = &scf_data.rimatr {
        // make direct call of ao2mo
        let rimatr = rimatr.0.to_rstsr_view(&device);
        let naux = rimatr.shape()[1];
        let mut cderi_xvo = unsafe { rt::empty(([naux, nvir, nocc].f(), &device)) };

        let mem_est = MemEstimate { batched: nao * nao + nao * ns + ns * nthreads, fixed: 0, thread: 0 };
        let mem_avail = scf_data.mol.ctrl.max_memory.map(|max_memory| max_memory - detect_used_memory_mb("proc"));
        let nbatch = calc_batch_size_from_mem_estimate::<f64>(&mem_est, mem_avail, None, true);
        timerecords.count_start("ao2mo");
        get_ao2mo_s2ij_to_s1_trans_with_output(
            rimatr.view(),
            Upper,
            &[vir_coeff.view()],
            &[occ_coeff.view()],
            Some(nbatch),
            &[cderi_xvo.view_mut()],
            |x| T::from_f64(x).unwrap(),
        );
        timerecords.count("ao2mo");
        cderi_xvo
    } else {
        // batched evaluate j3c with and decompose
        let mol_obj = &scf_data.mol;
        let mol = util::get_cint_mol(mol_obj);
        let aux = util::get_cint_aux(mol_obj);
        let naux = aux.nao();
        let mut j3c_xvo = unsafe { rt::empty(([naux, nvir, nocc].f(), &device)) };

        let mem_est = MemEstimate { batched: 2 * nao * nao + nao * ns + ns * nthreads, fixed: 0, thread: 0 };
        let mem_avail = scf_data.mol.ctrl.max_memory.map(|max_memory| max_memory - detect_used_memory_mb("proc"));
        let nbatch = calc_batch_size_from_mem_estimate::<f64>(&mem_est, mem_avail, None, true);

        timerecords.new_item("j2c", "generation and decomposition of 2c-2e ERI");
        timerecords.new_item("j3c", "batched 3c-2e ERI generation");
        timerecords.new_item("decomp", "solve 3c-2e ERI by decomposed 2c-2e ERI");

        timerecords.count_start("j2c");
        let j2c_decomp = pure_decompose::get_j2c_decomp(&aux, &device, scf_data.mol.ctrl.j2c_decomp);
        timerecords.count("j2c");

        let partition = aux.balance_partition(nbatch);
        let mut p0 = 0;
        for [sh0, sh1, np] in partition {
            let p1 = p0 + np;

            timerecords.count_start("j3c");
            let j3c = {
                let slc = [[0, mol.nbas()], [0, mol.nbas()], [sh0, sh1]];
                let (out, shape) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s2ij", slc).into();
                rt::asarray((out, shape.f(), &device))
            };
            timerecords.count("j3c");

            timerecords.count_start("ao2mo");
            let j3c_xvo_batch = j3c_xvo.i_mut((p0..p1, .., ..));
            get_ao2mo_s2ij_to_s1_trans_with_output(
                j3c.view(),
                Upper,
                &[vir_coeff.view()],
                &[occ_coeff.view()],
                Some(nbatch),
                &[j3c_xvo_batch],
                |x| T::from_f64(x).unwrap(),
            );
            timerecords.count("ao2mo");

            p0 = p1;
        }

        timerecords.count_start("decomp");
        // reshape to 2-d, make transpose to match the functionality, then reshape back
        let j3c_vox_2d = j3c_xvo.into_shape((naux, -1)).into_reverse_axes();
        let cderi_vox_2d = get_solved_j3c(j3c_vox_2d, &j2c_decomp, false);
        let cderi_xvo = cderi_vox_2d.into_reverse_axes().into_shape((naux, nvir, nocc));
        timerecords.count("decomp");

        cderi_xvo
    }
}

pub fn obtain_cderi_xvo_unrestricted<T>(
    scf_data: &SCF,
    timerecords: &mut TimeRecords,
    eigenvectors: Option<[&MatrixFull<f64>; 2]>,
    row_ranges: [Option<&[usize]>; 2],
    col_ranges: [Option<&[usize]>; 2],
) -> [Tsr<T>; 2]
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    const A: usize = 0;
    const B: usize = 1;

    let device = DeviceBLAS::default();

    // apply occ and vir orbital indices
    let mo_coeff = match eigenvectors {
        None => scf_data.eigenvectors.as_slice().to_rstsr(&device),
        Some(eigenvectors) => eigenvectors.as_slice().to_rstsr(&device),
    };
    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo;
    let nmo = mo_coeff.shape()[1];
    let occ_coeff = [
        match col_ranges[A] {
            None => {
                let list: Vec<usize> = (idx_core..idx_lumo[A]).collect();
                mo_coeff.i((.., .., A)).index_select(-1, &list)
            },
            Some(indices) => mo_coeff.i((.., .., A)).index_select(-1, indices),
        },
        match col_ranges[B] {
            None => {
                let list: Vec<usize> = (idx_core..idx_lumo[B]).collect();
                mo_coeff.i((.., .., B)).index_select(-1, &list)
            },
            Some(indices) => mo_coeff.i((.., .., B)).index_select(-1, indices),
        },
    ];
    let vir_coeff = [
        match row_ranges[A] {
            None => {
                let list: Vec<usize> = (idx_lumo[A]..nmo).collect();
                mo_coeff.i((.., .., A)).index_select(-1, &list)
            },
            Some(indices) => mo_coeff.i((.., .., A)).index_select(-1, indices),
        },
        match row_ranges[B] {
            None => {
                let list: Vec<usize> = (idx_lumo[B]..nmo).collect();
                mo_coeff.i((.., .., B)).index_select(-1, &list)
            },
            Some(indices) => mo_coeff.i((.., .., B)).index_select(-1, indices),
        },
    ];

    let nocc = [occ_coeff[A].shape()[1], occ_coeff[B].shape()[1]];
    let nvir = [vir_coeff[A].shape()[1], vir_coeff[B].shape()[1]];
    let nao = mo_coeff.shape()[0];
    let ns = nocc[A].max(nvir[A]).max(nocc[B]).max(nvir[B]);
    let nthreads = rayon::current_num_threads();

    if let Some(rimatr) = &scf_data.rimatr {
        // make direct call of ao2mo
        let rimatr = rimatr.0.to_rstsr_view(&device);
        let naux = rimatr.shape()[1];
        let mut cderi_xvo_a = unsafe { rt::empty(([naux, nvir[A], nocc[A]].f(), &device)) };
        let mut cderi_xvo_b = unsafe { rt::empty(([naux, nvir[B], nocc[B]].f(), &device)) };

        let mem_est = MemEstimate { batched: nao * nao + nao * ns + ns * nthreads, fixed: 0, thread: 0 };
        let mem_avail = scf_data.mol.ctrl.max_memory.map(|max_memory| max_memory - detect_used_memory_mb("proc"));
        let nbatch = calc_batch_size_from_mem_estimate::<f64>(&mem_est, mem_avail, None, true);
        timerecords.count_start("ao2mo");
        get_ao2mo_s2ij_to_s1_trans_with_output(
            rimatr.view(),
            Upper,
            &[vir_coeff[A].view(), vir_coeff[B].view()],
            &[occ_coeff[A].view(), occ_coeff[B].view()],
            Some(nbatch),
            &[cderi_xvo_a.view_mut(), cderi_xvo_b.view_mut()],
            |x| T::from_f64(x).unwrap(),
        );
        timerecords.count("ao2mo");
        [cderi_xvo_a, cderi_xvo_b]
    } else {
        // batched evaluate j3c with and decompose
        let mol_obj = &scf_data.mol;
        let mol = util::get_cint_mol(mol_obj);
        let aux = util::get_cint_aux(mol_obj);
        let naux = aux.nao();
        let mem_est = MemEstimate {
            batched: 2 * nao * nao + nao * ns + ns * nthreads,
            fixed: nocc[A] * nvir[A] * naux + nocc[B] * nvir[B] * naux,
            thread: 0,
        };
        let mem_avail = scf_data.mol.ctrl.max_memory.map(|max_memory| max_memory - detect_used_memory_mb("proc"));
        let nbatch = calc_batch_size_from_mem_estimate::<f64>(&mem_est, mem_avail, None, true);

        let mut j3c_xvo_a = unsafe { rt::empty(([naux, nvir[A], nocc[A]].f(), &device)) };
        let mut j3c_xvo_b = unsafe { rt::empty(([naux, nvir[B], nocc[B]].f(), &device)) };

        timerecords.new_item("j2c", "generation and decomposition of 2c-2e ERI");
        timerecords.new_item("j3c", "batched 3c-2e ERI generation");
        timerecords.new_item("decomp", "solve 3c-2e ERI by decomposed 2c-2e ERI");

        timerecords.count_start("j2c");
        let j2c_decomp = pure_decompose::get_j2c_decomp(&aux, &device, scf_data.mol.ctrl.j2c_decomp);
        timerecords.count("j2c");

        let partition = aux.balance_partition(nbatch);
        let mut p0 = 0;
        for [sh0, sh1, np] in partition {
            let p1 = p0 + np;

            timerecords.count_start("j3c");
            let j3c = {
                let slc = [[0, mol.nbas()], [0, mol.nbas()], [sh0, sh1]];
                let (out, shape) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s2ij", slc).into();
                rt::asarray((out, shape.f(), &device))
            };
            timerecords.count("j3c");

            timerecords.count_start("ao2mo");
            let j3c_xvo_batch = [j3c_xvo_a.i_mut((p0..p1, .., ..)), j3c_xvo_b.i_mut((p0..p1, .., ..))];
            get_ao2mo_s2ij_to_s1_trans_with_output(
                j3c.view(),
                Upper,
                &[vir_coeff[A].view(), vir_coeff[B].view()],
                &[occ_coeff[A].view(), occ_coeff[B].view()],
                Some(nbatch),
                &j3c_xvo_batch,
                |x| T::from_f64(x).unwrap(),
            );
            timerecords.count("ao2mo");

            p0 = p1;
        }

        timerecords.count_start("decomp");
        // reshape to 2-d, make transpose to match the functionality, then reshape back
        let j3c_vox_a_2d = j3c_xvo_a.into_shape((naux, -1)).into_reverse_axes();
        let j3c_vox_b_2d = j3c_xvo_b.into_shape((naux, -1)).into_reverse_axes();
        let cderi_vox_a_2d = get_solved_j3c(j3c_vox_a_2d, &j2c_decomp, false);
        let cderi_vox_b_2d = get_solved_j3c(j3c_vox_b_2d, &j2c_decomp, false);
        let cderi_xvo_a = cderi_vox_a_2d.into_reverse_axes().into_shape((naux, nvir[A], nocc[A]));
        let cderi_xvo_b = cderi_vox_b_2d.into_reverse_axes().into_shape((naux, nvir[B], nocc[B]));
        timerecords.count("decomp");

        [cderi_xvo_a, cderi_xvo_b]
    }
}
