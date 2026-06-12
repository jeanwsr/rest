#![warn(unused)]
use crate::ri_jk;
use crate::scf_io::SCF;
use crate::utilities::rstsr_util::*;
use crate::utilities::TimeRecords;
use num::FromPrimitive;
use rstsr::prelude::*;

use rt::blas::{BlasFloat, LapackDriverAPI};

pub fn evaluate_ript2_eng<T>(
    scf_data: &SCF,
    timerecords: &mut TimeRecords,
    occidx: [Option<&[usize]>; 2],
    viridx: [Option<&[usize]>; 2],
) -> [f64; 3]
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = DeviceBLAS::default();

    let mo_energy = (&scf_data.eigenvalues[0]).to_rstsr(&device);
    let mo_occupation = (&scf_data.occupation[0]).to_rstsr(&device);

    // apply occ and vir orbital indices
    // note occupation in restricted case should be divided by 2
    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo[0];
    let num_mo = mo_energy.size();

    // build occupied and virtual orbital index lists
    // when None, default to the conventional contiguous range
    let occ_list: Vec<usize> = occidx[0]
        .map(|x| x.to_vec())
        .unwrap_or_else(|| (idx_core..idx_lumo).collect());
    let vir_list: Vec<usize> = viridx[0]
        .map(|x| x.to_vec())
        .unwrap_or_else(|| (idx_lumo..num_mo).collect());

    let occ_energy = mo_energy.index_select(-1, &occ_list);
    let vir_energy = mo_energy.index_select(-1, &vir_list);
    let occ_occupation = mo_occupation.index_select(-1, &occ_list) / 2;
    let vir_occupation = mo_occupation.index_select(-1, &vir_list) / 2;

    // perform ao2mo
    let cderi_xvo = ri_jk::obtain_cderi_xvo_restricted(
        scf_data, timerecords, None,
        Some(vir_list.as_slice()),
        Some(occ_list.as_slice()),
    );

    timerecords.count_start("c_r5dft");
    let pair_eng = super::pure_pt2_pair_eng::get_ript2_energy_pair_intra(
        cderi_xvo.view(),
        occ_energy.view(),
        vir_energy.view(),
        Some(occ_occupation.view()),
        Some(vir_occupation.view()),
        false,
    );
    timerecords.count("c_r5dft");

    let eng_os = pair_eng.os.unwrap().sum();
    let eng_ss = pair_eng.ss.unwrap().sum();
    let eng_tot = eng_os + eng_ss;
    [eng_tot, eng_os, eng_ss]
}

pub fn evaluate_riupt2_eng<T>(
    scf_data: &SCF,
    timerecords: &mut TimeRecords,
    occidx: [Option<&[usize]>; 2],
    viridx: [Option<&[usize]>; 2],
) -> [f64; 3]
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    const A: usize = 0;
    const B: usize = 1;

    let device = DeviceBLAS::default();

    let mo_energy = scf_data.eigenvalues.as_slice().to_rstsr(&device);
    let mo_occupation = scf_data.occupation.as_slice().to_rstsr(&device);

    // apply occ and vir orbital indices
    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo;
    let num_mo = mo_energy.shape()[0];

    // build occupied and virtual orbital index lists per spin
    // when None, default to the conventional contiguous range
    let occ_lists: [Vec<usize>; 2] = [A, B].map(|spin| {
        occidx[spin]
            .map(|x| x.to_vec())
            .unwrap_or_else(|| (idx_core..idx_lumo[spin]).collect())
    });
    let vir_lists: [Vec<usize>; 2] = [A, B].map(|spin| {
        viridx[spin]
            .map(|x| x.to_vec())
            .unwrap_or_else(|| (idx_lumo[spin]..num_mo).collect())
    });

    // slice each spin channel to 1D then index_select
    let occ_energy = [
        mo_energy.i((.., A)).index_select(-1, &occ_lists[A]),
        mo_energy.i((.., B)).index_select(-1, &occ_lists[B]),
    ];
    let vir_energy = [
        mo_energy.i((.., A)).index_select(-1, &vir_lists[A]),
        mo_energy.i((.., B)).index_select(-1, &vir_lists[B]),
    ];
    let occ_occupation = [
        mo_occupation.i((.., A)).index_select(-1, &occ_lists[A]),
        mo_occupation.i((.., B)).index_select(-1, &occ_lists[B]),
    ];
    let vir_occupation = [
        mo_occupation.i((.., A)).index_select(-1, &vir_lists[A]),
        mo_occupation.i((.., B)).index_select(-1, &vir_lists[B]),
    ];

    // perform ao2mo
    let cderi_xvo = ri_jk::obtain_cderi_xvo_unrestricted(
        scf_data, timerecords, None,
        [Some(vir_lists[A].as_slice()), Some(vir_lists[B].as_slice())],
        [Some(occ_lists[A].as_slice()), Some(occ_lists[B].as_slice())],
    );

    timerecords.count_start("c_r5dft");
    let pair_eng_aa = super::pure_pt2_pair_eng::get_ript2_energy_pair_intra(
        cderi_xvo[A].view(),
        occ_energy[A].view(),
        vir_energy[A].view(),
        Some(occ_occupation[A].view()),
        Some(vir_occupation[A].view()),
        true,
    );
    let pair_eng_bb = super::pure_pt2_pair_eng::get_ript2_energy_pair_intra(
        cderi_xvo[B].view(),
        occ_energy[B].view(),
        vir_energy[B].view(),
        Some(occ_occupation[B].view()),
        Some(vir_occupation[B].view()),
        true,
    );
    let pair_eng_ab = super::pure_pt2_pair_eng::get_riuospt2_energy_pair_intra(
        [cderi_xvo[A].view(), cderi_xvo[B].view()],
        [occ_energy[A].view(), occ_energy[B].view()],
        [vir_energy[A].view(), vir_energy[B].view()],
        Some([occ_occupation[A].view(), occ_occupation[B].view()]),
        Some([vir_occupation[A].view(), vir_occupation[B].view()]),
    );
    timerecords.count("c_r5dft");

    let eng_os = pair_eng_ab.os.unwrap().sum();
    let eng_ss = 0.25 * (pair_eng_aa.ss.unwrap().sum() + pair_eng_bb.ss.unwrap().sum());
    let eng_tot = eng_os + eng_ss;
    [eng_tot, eng_os, eng_ss]
}
