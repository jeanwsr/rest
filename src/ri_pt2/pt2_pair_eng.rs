#![warn(unused)]
use crate::ri_jk;
use crate::scf_io::SCF;
use crate::utilities::rstsr_util::*;
use crate::utilities::TimeRecords;
use num::FromPrimitive;
use rstsr::prelude::*;

use rt::blas::{BlasFloat, LapackDriverAPI};

pub fn evaluate_ript2_eng<T>(scf_data: &SCF, timerecords: &mut TimeRecords) -> [f64; 3]
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
    let occ_energy = mo_energy.i(idx_core..idx_lumo);
    let vir_energy = mo_energy.i(idx_lumo..);
    let occ_occupation = mo_occupation.i(idx_core..idx_lumo) / 2;
    let vir_occupation = mo_occupation.i(idx_lumo..) / 2;

    // perform ao2mo
    let cderi_xvo = ri_jk::obtain_cderi_xvo_restricted(scf_data, timerecords, None, None, None);

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

pub fn evaluate_riupt2_eng<T>(scf_data: &SCF, timerecords: &mut TimeRecords) -> [f64; 3]
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
    let occ_energy = [mo_energy.i((idx_core..idx_lumo[A], A)), mo_energy.i((idx_core..idx_lumo[B], B))];
    let vir_energy = [mo_energy.i((idx_lumo[A].., A)), mo_energy.i((idx_lumo[B].., B))];
    let occ_occupation = [mo_occupation.i((idx_core..idx_lumo[A], A)), mo_occupation.i((idx_core..idx_lumo[B], B))];
    let vir_occupation = [mo_occupation.i((idx_lumo[A].., A)), mo_occupation.i((idx_lumo[B].., B))];

    // perform ao2mo
    let cderi_xvo = ri_jk::obtain_cderi_xvo_unrestricted(scf_data, timerecords, None, None, None);

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
