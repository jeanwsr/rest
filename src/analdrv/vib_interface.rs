use super::prelude::*;
use crate::geom_io::get_mass_charge;
use crate::analdrv::vib::*;
use crate::ri_jk::util::get_cint_mol;
use crate::SCF;

/// Vibrational analysis interface for REST.
///
/// `de_hess` should be of shape `[3, 3, natm, natm]`, and is the hessian matrix in atomic units.
pub fn vibration_analysis_interface(
    scf_data: &SCF,
    config: &AnalDrvConfig,
    de_hess: TsrView,
) -> (Vec<f64>, VibInfo, Option<GauThermoInfo>) {
    let device = de_hess.device().clone();
    let mol_obj = &scf_data.mol;
    let mol = get_cint_mol(mol_obj);

    // --- vibrational analysis --- //

    println!("=============== Vibrational Analysis (in analdrv) ===============");

    // first transpose hessian to [3*natm, 3*natm]
    let natm = de_hess.shape()[3];
    let hess = de_hess.transpose((0, 2, 1, 3)).into_shape((3 * natm, 3 * natm));
    let atm_list = config.nucgrad.atm_list.clone().unwrap_or_else(|| (0..mol.natm()).collect_vec());
    let elems = atm_list.iter().map(|&i| scf_data.mol.geom.elem[i].clone()).collect_vec();

    let mass_charge = get_mass_charge(&elems);
    let mass = mass_charge.iter().map(|(m, _)| *m).collect_vec();
    let mass_rt = rt::asarray((&mass, &device));

    let geom = atm_list.iter().map(|&i| mol.atom_coord(i)).collect_vec();
    let geom_rt = rt::asarray((&geom, &device)).into_unpack_array(0);
    let vib = harmonic_analysis(hess.view(), geom_rt.view(), mass_rt.view(), true, true);
    println!("");
    let elems_ref = elems.iter().map(|e| e.as_str()).collect_vec();
    let msg_vib = print_vibs(&vib, &elems_ref, NormCo::X, true, Some(3), 4, None);
    println!("{}", msg_vib);
    println!("");

    println!("=============== End of Vibrational Analysis (in analdrv) ===============");
    println!("");

    // --- thermo analysis (gaussian-style) --- //

    // this is activated by `gau_thermo = true` in control input, and not activated by default.

    let gau_th = config.nucgrad.gau_thermo.then(|| {

        println!("=============== Thermo Analysis (Usual Style in analdrv) ===============");
        println!("");
        println!("Note: This is gaussian-style thermo analysis.");

        let geom_c = mass_centred_geom(geom_rt.view(), mass_rt.view());
        let rc_cm = rotation_const(mass_rt.view(), geom_c.view(), "wavenumber").to_vec();
        let rc_ghz = rotation_const(mass_rt.view(), geom_c.view(), "GHz");
        let rotor = RotorType::from_rot_const_ghz(rc_ghz.view());
        let mass_sum = mass.iter().sum::<f64>();
        let e0 = scf_data.scf_energy;
        let multiplicity = scf_data.mol.ctrl.spin;

        use super::point_group_detect::interface_to_rest::get_full_point_group_for_vib;
        let tol_pg = config.nucgrad.tol_point_group / (1.0 + natm as f64).sqrt();
        let (pg_name, pg_sigma) = get_full_point_group_for_vib(&elems, &mass, &geom, tol_pg);

        let thermo_ctrl = scf_data.mol.ctrl.thermo.as_ref();
        let temperature = thermo_ctrl.map(|t| t.temperature).unwrap_or(298.15);
        let pressure = thermo_ctrl.map(|t| t.pressure * 101325.0).unwrap_or(101325.0);
        // Electronic energy can be zero, which means we use SCF energy as electronic energy.
        let mut electronic_energy = thermo_ctrl.map(|t| t.electronic_energy).unwrap_or(e0);
        if electronic_energy == 0.0 {
            electronic_energy = e0;
        }
        
        // if control input has symmetry number, use it; otherwise, use point group detected sigma.
        let mut symmetry_number = thermo_ctrl.map(|t| t.symmetry_number as i64).unwrap_or(pg_sigma as i64);
        if symmetry_number <= 0 {
            symmetry_number = pg_sigma as i64;
        }

        let th = thermo(
            &vib,
            temperature,
            pressure,
            multiplicity as _,
            mass_sum,
            electronic_energy,
            symmetry_number,
            &rc_cm,
            rotor,
        );
        println!("Point group: {}, sigma (rotation symmetry number): {}", pg_name, pg_sigma);
        let msg_th = print_gau_thermo(&th, multiplicity as _, mass_sum);
        println!("{}", msg_th);
        println!("");

        println!("=============== End of Thermo Analysis (Usual Style in analdrv) ===============");
        th
    });

    (de_hess.into_shape(-1).into_vec(), vib, gau_th)
}
