#![warn(unused)]
use super::rhf::*;
use crate::external_field::extfield::ExtFieldGrad;
use crate::grad::traits::GradAPI;
use crate::ri_jk;
use crate::scf_io::{self, SCF};
use crate::utilities::memory_batch::*;
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use tensors::MatrixFull;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;

/// Gradient structure and values for UHF method.
pub struct RIUHFGradient<'a> {
    pub scf_data: &'a SCF,
    pub flags: RIHFGradientFlags,
    pub result: HashMap<String, MatrixFull<f64>>,
}

impl GradAPI for RIUHFGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result.get("de").unwrap().clone()
    }

    fn get_energy(&self) -> f64 {
        self.scf_data.scf_energy
    }
}

impl RIUHFGradient<'_> {
    pub fn new(scf_data: &SCF) -> RIUHFGradient<'_> {
        // check SCF type
        match scf_data.scftype {
            scf_io::SCFType::UHF => {},
            _ => panic!("SCFtype is not sutiable for UHF gradient."),
        };

        let flags = build_ri_jk_grad_flags(scf_data);
        RIUHFGradient { scf_data, flags, result: HashMap::new() }
    }

    /// Derivatives of nuclear repulsion energy with reference to nuclear coordinates
    pub fn calc_de_nuc(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let de_nuc = calc_de_nuc(mol);
        self.result.insert("de_nuc".into(), de_nuc);
        return self;
    }

    pub fn calc_de_ovlp(&mut self) -> &mut Self {
        // preparation
        let mol_obj = &self.scf_data.mol;
        let mol = ri_jk::util::get_cint_mol(mol_obj);
        let device = DeviceBLAS::default();

        // orbital informations
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let mo_energy = get_mo_energy(self.scf_data, &device);

        // tsr_int1e_ipovlp
        let tsr_int1e_ipovlp = {
            let (out, shape) = mol.integrate("int1e_ipovlp", "s1", None).into();
            rt::asarray((out, shape, &device))
        };

        let mut dme0 = get_dme0(mo_coeff[0].view(), mo_occ[0].view(), mo_energy[0].view());
        dme0 += get_dme0(mo_coeff[1].view(), mo_occ[1].view(), mo_energy[1].view());
        let dao_ovlp = get_grad_dao_ovlp(tsr_int1e_ipovlp.view(), dme0.view());

        // de_ovlp
        let natm = mol_obj.geom.elem.len();
        let mut de_ovlp = rt::zeros(([3, natm], &device));
        let ao_slice = mol_obj.aoslice_by_atom();

        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm];
            *&mut de_ovlp.i_mut((.., atm)) += dao_ovlp.i(p0..p1).sum_axes(0);
        }

        // note f-contiguous transpose
        let de_ovlp_raw = de_ovlp.into_shape(-1).into_raw();
        let de_ovlp = MatrixFull::from_vec([3, natm], de_ovlp_raw).unwrap();
        self.result.insert("de_ovlp".into(), de_ovlp);
        return self;
    }

    pub fn calc_de_hcore(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let natm = mol.geom.elem.len();
        let device = DeviceBLAS::default();

        let dm = get_dm(self.scf_data, &device);
        let dm = &dm[0] + &dm[1];
        let mut de_hcore: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut gen_deriv_hcore = generator_deriv_hcore(&self.scf_data);
        for atm in 0..natm {
            *&mut de_hcore.i_mut((.., atm)) += (gen_deriv_hcore(atm) * &dm).sum_axes([0, 1]);
        }

        let de_hcore_raw = de_hcore.into_shape(-1).into_raw();
        let de_hcore = MatrixFull::from_vec([3, natm], de_hcore_raw).unwrap();
        self.result.insert("de_hcore".into(), de_hcore);
        return self;
    }

    pub fn calc_de_qmmm(&mut self) -> &mut Self {
        let scf_data = &self.scf_data;
        let mol = &scf_data.mol;
        let num_qm_atoms = mol.geom.elem.len();

        let mut de_qmmm = MatrixFull::new([3, num_qm_atoms], 0.0);

        if mol.geom.ghost_pc_chrg.is_empty() {
            self.result.insert("de_qmmm".into(), de_qmmm);
            return self;
        }

        let num_ghosts = mol.geom.ghost_pc_chrg.len();
        let ghost_pc_chrg = &mol.geom.ghost_pc_chrg;
        let ghost_pc_pos = &mol.geom.ghost_pc_pos;
        let qm_position = &mol.geom.position;
        let qm_elem = &mol.geom.elem;
        let aoslices = mol.aoslice_by_atom();
        let nao = if let Some(last) = aoslices.last() { last[3] } else { 0 };

        let nuclear_charges = crate::geom_io::get_charge(qm_elem);
        for a in 0..num_qm_atoms {
            let pos_a = [qm_position[[0, a]], qm_position[[1, a]], qm_position[[2, a]]];
            for i in 0..num_ghosts {
                let pos_x = [ghost_pc_pos[[0, i]], ghost_pc_pos[[1, i]], ghost_pc_pos[[2, i]]];
                let r_vec = [pos_a[0] - pos_x[0], pos_a[1] - pos_x[1], pos_a[2] - pos_x[2]];
                let r_sq = r_vec.iter().map(|&x| x * x).sum::<f64>();
                if r_sq > 1e-12 {
                    let prefactor = -1.0 * (nuclear_charges[a] * ghost_pc_chrg[i]) / (r_sq * r_sq.sqrt());
                    for t in 0..3 {
                        de_qmmm[[t, a]] += prefactor * r_vec[t];
                    }
                }
            }
        }

        let mut deriv_hcore = vec![0.0; 3 * nao * nao];

        let mut temp_mol = mol.clone();

        for i in 0..num_ghosts {
            let q_i = ghost_pc_chrg[i];
            let pos_i = [ghost_pc_pos[[0, i]], ghost_pc_pos[[1, i]], ghost_pc_pos[[2, i]]];

            temp_mol.with_rinv_origin(pos_i, |mol_mut| {
                let cint = mol_mut.initialize_cint(false);
                let iprinv_out = cint.integrate("int1e_iprinv", "s1", None);

                if let Some(out_vec) = iprinv_out.out {
                    for t in 0..3 {
                        for nu in 0..nao {
                            for mu in 0..nao {
                                let idx = mu + nu * nao + t * (nao * nao);
                                if let Some(&val) = out_vec.get(idx) {
                                    deriv_hcore[t * nao * nao + mu * nao + nu] += q_i * val;
                                }
                            }
                        }
                    }
                }
            });
        }

        let is_sp = mol.ctrl.spin_polarization;
        let dm = &scf_data.density_matrix;
        let use_double_dm = is_sp && dm.len() > 1 && dm[1].size == [nao, nao];

        for a in 0..num_qm_atoms {
            let [_, _, p0, p1] = aoslices[a];
            for t in 0..3 {
                let mut sum_val = 0.0;
                for mu in p0..p1 {
                    for nu in 0..nao {
                        let h_val = deriv_hcore[t * nao * nao + mu * nao + nu];
                        let dm_val = if !use_double_dm { dm[0][[mu, nu]] } else { dm[0][[mu, nu]] + dm[1][[mu, nu]] };
                        sum_val += h_val * dm_val;
                    }
                }
                de_qmmm[[t, a]] += 2.0 * sum_val;
            }
        }
        self.result.insert("de_qmmm".into(), de_qmmm);
        return self;
    }

    pub fn calc_de_ext_field(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let natm = mol.geom.elem.len();

        if self.flags.ext_field_dipole.is_none() {
            self.result.insert("de_ext_field".into(), MatrixFull::new([3, natm], 0.0));
            return self;
        }

        let dm = get_dm(self.scf_data, &DeviceBLAS::default());
        let dm = &dm[0] + &dm[1];
        let shape = dm.shape().to_vec().try_into().unwrap();
        let dm = MatrixFull::from_vec(shape, dm.into_shape(-1).into_vec()).unwrap();
        let mut ext_grad = ExtFieldGrad::new(mol, &dm);
        let de_ext = ext_grad.calc();
        self.result.insert("de_ext_field".into(), de_ext);
        return self;
    }

    pub fn calc_de_jk(&mut self) -> &mut Self {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("de-jk prepr 1", "de-jk prepr 1 (basic setup)");
        time_records.new_item("de-jk prepr 2", "de-jk prepr 2 (itm setup, hyb)");
        time_records.new_item("de-jk batch int 1", "de-jk batch int (int3c2e_ip1, int3c2e_ip2, hyb)");
        time_records.new_item("de-jk batch 1", "de-jk 1 batch (get_grad_dao_j_int3c2e_ip1)");
        time_records.new_item("de-jk batch 2", "de-jk 2 batch (get_grad_daux_j_int3c2e_ip2)");
        time_records.new_item("de-jk batch 3", "de-jk 3 batch (get_itm_k_ao, hyb)");
        time_records.new_item("de-jk batch 4", "de-jk 4 batch (get_grad_dao_k_int3c2e_ip1, hyb)");
        time_records.new_item("de-jk batch 5", "de-jk 5 batch (get_grad_daux_k_int3c2e_ip2, hyb)");
        time_records.new_item("de-jk prepr 3", "de-jk prepr 3 (itm setup, rsh)");
        time_records.new_item("de-jk batch int 2", "de-jk batch int (int3c2e_ip1, int3c2e_ip2, rsh)");
        time_records.new_item("de-jk batch 6", "de-jk 6 batch (get_itm_k_ao, rsh)");
        time_records.new_item("de-jk batch 7", "de-jk 7 batch (get_grad_dao_k_int3c2e_ip1, rsh)");
        time_records.new_item("de-jk batch 8", "de-jk 8 batch (get_grad_daux_k_int3c2e_ip2, rsh)");

        time_records.count_start("de-jk prepr 1");

        let mol_obj = &self.scf_data.mol;
        let natm = mol_obj.geom.elem.len();
        let mol = ri_jk::util::get_cint_mol(mol_obj);
        let aux = ri_jk::util::get_cint_aux(mol_obj);
        let device = DeviceBLAS::default();

        // density matrix and triu-packed density matrix
        let dm = get_dm(self.scf_data, &device);
        let dm = &dm[0] + &dm[1];
        let dm_tp = pack_triu_tilde(dm.view());

        // orbital coefficients
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let nao = dm.shape()[0];
        let nocc = [
            mo_occ[0].iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum::<usize>(),
            mo_occ[1].iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum::<usize>(),
        ];
        let occ_coeff = [mo_coeff[0].i((.., ..nocc[0])), mo_coeff[1].i((.., ..nocc[1]))];

        // eigen-decomposed ERI
        let ederi_utp = {
            let msg = "Decomposed ERI not found, possibly due to insufficient memory. We do not support ri-direct gradient calculation.";
            let tsr = self.scf_data.rimatr.as_ref().expect(msg);
            rt::asarray((&tsr.0.data, tsr.0.size, &device))
        };
        let naux = ederi_utp.shape()[1];

        // tsr_int2c2e_l: J^-1/2
        let j2c_decomp_option = self.scf_data.mol.ctrl.j2c_decomp;
        let j2c_decomp = ri_jk::get_j2c_decomp(&aux, &device, j2c_decomp_option);

        // tsr_int2c2e_ip1
        let tsr_int2c2e_ip1 = {
            let (out, shape) = aux.integrate("int2c2e_ip1", "s1", None).into();
            rt::asarray((out, shape, &device))
        };

        // shell partition of int3c2e
        let aux_loc = aux.ao_loc();

        // available memory in MB, if not set, will be calculated from system
        let mem_avail = self.flags.max_memory.map(|max_memory| max_memory - detect_used_memory_mb("proc"));
        let aux_batch_size = calc_batch_size::<f64>(
            8 * nao * nao,
            mem_avail,
            None,
            Some(naux * (nocc[0] * nocc[0] + nocc[1] * nocc[1])),
        );
        let aux_batch_size = aux_batch_size.min(216);
        let aux_partition = blocksize_partition(&aux_loc, aux_batch_size);

        time_records.count("de-jk prepr 1");

        // basic setup finished
        // begin hybrid computation

        time_records.count_start("de-jk prepr 2");

        // temporaries for de_jaux, de_kaux
        let mut itm_j = rt::full(([], f64::NAN, &device));
        let mut dao_j = rt::full(([], f64::NAN, &device));
        let mut daux_j = rt::full(([], f64::NAN, &device));
        if self.flags.factor_j.is_some() {
            itm_j = get_itm_j(&j2c_decomp, ederi_utp.view(), dm_tp.view());
            dao_j = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
            daux_j = get_grad_daux_j_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_j.view());
        }

        let mut itm_k_occtp = [rt::full(([], f64::NAN, &device)), rt::full(([], f64::NAN, &device))];
        let mut dao_k = rt::full(([], f64::NAN, &device));
        let mut daux_k = rt::full(([], f64::NAN, &device));
        if self.flags.factor_k.is_some() {
            itm_k_occtp = [
                get_itm_k_occtp(&j2c_decomp, ederi_utp.view(), occ_coeff[0].view()),
                get_itm_k_occtp(&j2c_decomp, ederi_utp.view(), occ_coeff[1].view()),
            ];
            dao_k = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
            let itm_k_aux = get_itm_k_aux(itm_k_occtp[0].view_mut()) + get_itm_k_aux(itm_k_occtp[1].view_mut());
            daux_k = get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_k_aux.view());
        }

        time_records.count("de-jk prepr 2");

        let mut idx_aux_start = 0;
        for [shl0, shl1] in aux_partition.clone() {
            let shl_naux = aux_loc[shl1] - aux_loc[shl0];
            let shl_slices = [[0, mol.nbas()], [0, mol.nbas()], [shl0, shl1]];
            let (p0, p1) = (idx_aux_start, idx_aux_start + shl_naux);

            time_records.count_start("de-jk batch int 1");
            // int3c2e_ip1
            let tsr_int3c2e_ip1 = {
                let (out, shape) = CInt::integrate_cross("int3c2e_ip1", [&mol, &mol, &aux], "s1", shl_slices).into();
                rt::asarray((out, shape, &device))
            };

            // int3c2e_ip2
            let mut tsr_int3c2e_ip2 = rt::full(([], f64::NAN, &device));
            if self.flags.auxbasis_response {
                tsr_int3c2e_ip2 = {
                    let (out, shape) =
                        CInt::integrate_cross("int3c2e_ip2", [&mol, &mol, &aux], "s2ij", shl_slices).into();
                    rt::asarray((out, shape, &device))
                };
            }
            time_records.count("de-jk batch int 1");

            if self.flags.factor_j.is_some() {
                time_records.count_start("de-jk batch 1");
                *&mut dao_j += get_grad_dao_j_int3c2e_ip1(tsr_int3c2e_ip1.view(), dm.view(), itm_j.i(p0..p1));
                time_records.count("de-jk batch 1");

                if self.flags.auxbasis_response {
                    time_records.count_start("de-jk batch 2");
                    *&mut daux_j.i_mut(p0..p1) +=
                        get_grad_daux_j_int3c2e_ip2(tsr_int3c2e_ip2.view(), dm_tp.view(), itm_j.i(p0..p1));
                    time_records.count("de-jk batch 2");
                }
            }

            if self.flags.factor_k.is_some() {
                time_records.count_start("de-jk batch 3");
                let itm_k_ao = get_itm_k_ao(itm_k_occtp[0].i((.., p0..p1)), occ_coeff[0].view())
                    + get_itm_k_ao(itm_k_occtp[1].i((.., p0..p1)), occ_coeff[1].view());
                time_records.count("de-jk batch 3");

                time_records.count_start("de-jk batch 4");
                *&mut dao_k += get_grad_dao_k_int3c2e_ip1(tsr_int3c2e_ip1.view(), itm_k_ao.view());
                time_records.count("de-jk batch 4");

                if self.flags.auxbasis_response {
                    time_records.count_start("de-jk batch 5");
                    *&mut daux_k.i_mut(p0..p1) += get_grad_daux_k_int3c2e_ip2(tsr_int3c2e_ip2.view(), itm_k_ao.view());
                    time_records.count("de-jk batch 5");
                }
            }

            idx_aux_start += shl_naux;
        }

        // begin rsh computation (evaluate short-range part of K, so negative omega for libcint)

        let mut dao_sr = rt::full(([], f64::NAN, &device));
        let mut daux_sr = rt::full(([], f64::NAN, &device));

        if let Some(omega) = self.flags.omega {
            // setup molecules for short-range integrals
            let (mut mol, mut aux) = (mol.clone(), aux.clone());
            mol.set_omega(-omega);
            aux.set_omega(-omega);

            // the already existed decomposed ERI
            let ederi_utp_rimatr_sr = self.scf_data.rimatr_sr.as_ref().unwrap();
            let ederi_utp_sr = {
                let tsr = ederi_utp_rimatr_sr;
                rt::asarray((&tsr.0.data, tsr.0.size, &device))
            };

            // regenerate essential cheap integrals
            let j2c_decomp_option = self.scf_data.mol.ctrl.j2c_decomp;
            let j2c_decomp_sr = ri_jk::get_j2c_decomp(&aux, &device, j2c_decomp_option);

            let tsr_int2c2e_ip1 = {
                let (out, shape) = aux.integrate("int2c2e_ip1", "s1", None).into();
                rt::asarray((out, shape, &device))
            };

            time_records.count_start("de-jk prepr 3");

            // temporaries for de_sraux
            let mut itm_k_occtp_sr = [
                get_itm_k_occtp(&j2c_decomp_sr, ederi_utp_sr.view(), occ_coeff[0].view()),
                get_itm_k_occtp(&j2c_decomp_sr, ederi_utp_sr.view(), occ_coeff[1].view()),
            ];
            dao_sr = rt::zeros(([nao, 3], &device));
            let itm_sr_aux = get_itm_k_aux(itm_k_occtp_sr[0].view_mut()) + get_itm_k_aux(itm_k_occtp_sr[1].view_mut());
            daux_sr = get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_sr_aux.view());

            time_records.count("de-jk prepr 3");

            let mut idx_aux_start = 0;
            for [shl0, shl1] in aux_partition.clone() {
                let shl_naux = aux_loc[shl1] - aux_loc[shl0];
                let shl_slices = [[0, mol.nbas()], [0, mol.nbas()], [shl0, shl1]];
                let (p0, p1) = (idx_aux_start, idx_aux_start + shl_naux);

                time_records.count_start("de-jk batch int 2");
                // int3c2e_ip1
                let tsr_int3c2e_ip1 = {
                    let (out, shape) =
                        CInt::integrate_cross("int3c2e_ip1", [&mol, &mol, &aux], "s1", shl_slices).into();
                    rt::asarray((out, shape, &device))
                };

                // int3c2e_ip2
                let mut tsr_int3c2e_ip2 = rt::full(([], f64::NAN, &device));
                if self.flags.auxbasis_response {
                    tsr_int3c2e_ip2 = {
                        let (out, shape) =
                            CInt::integrate_cross("int3c2e_ip2", [&mol, &mol, &aux], "s2ij", shl_slices).into();
                        rt::asarray((out, shape, &device))
                    };
                }
                time_records.count("de-jk batch int 2");

                time_records.count_start("de-jk batch 6");
                let itm_sr_ao = get_itm_k_ao(itm_k_occtp_sr[0].i((.., p0..p1)), occ_coeff[0].view())
                    + get_itm_k_ao(itm_k_occtp_sr[1].i((.., p0..p1)), occ_coeff[1].view());
                time_records.count("de-jk batch 6");

                time_records.count_start("de-jk batch 7");
                *&mut dao_sr += get_grad_dao_k_int3c2e_ip1(tsr_int3c2e_ip1.view(), itm_sr_ao.view());
                time_records.count("de-jk batch 7");

                if self.flags.auxbasis_response {
                    time_records.count_start("de-jk batch 8");
                    *&mut daux_sr.i_mut(p0..p1) +=
                        get_grad_daux_k_int3c2e_ip2(tsr_int3c2e_ip2.view(), itm_sr_ao.view());
                    time_records.count("de-jk batch 8");
                }

                idx_aux_start += shl_naux;
            }
        }

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        // de_j, de_k, de_sr, de_jaux, de_kaux, de_sraux
        let mut de_j = rt::full(([3, natm], f64::NAN, &device));
        let mut de_k = rt::full(([3, natm], f64::NAN, &device));
        let mut de_sr = rt::full(([3, natm], f64::NAN, &device));
        let mut de_jaux = rt::full(([3, natm], f64::NAN, &device));
        let mut de_kaux = rt::full(([3, natm], f64::NAN, &device));
        let mut de_sraux = rt::full(([3, natm], f64::NAN, &device));
        let ao_slice = mol_obj.aoslice_by_atom();
        let aux_slice = mol_obj.make_auxmol_fake().aoslice_by_atom();

        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm].clone().try_into().unwrap();
            if self.flags.factor_j.is_some() {
                *&mut de_j.i_mut((.., atm)).assign(dao_j.i(p0..p1).sum_axes(0));
            }
            if self.flags.factor_k.is_some() {
                *&mut de_k.i_mut((.., atm)).assign(dao_k.i(p0..p1).sum_axes(0));
            }
            if self.flags.omega.is_some() {
                *&mut de_sr.i_mut((.., atm)).assign(dao_sr.i(p0..p1).sum_axes(0));
            }

            let [_, _, p0, p1] = aux_slice[atm].clone().try_into().unwrap();
            if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
                *&mut de_jaux.i_mut((.., atm)).assign(daux_j.i(p0..p1).sum_axes(0));
            }
            if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
                *&mut de_kaux.i_mut((.., atm)).assign(daux_k.i(p0..p1).sum_axes(0));
            }
            if self.flags.omega.is_some() && self.flags.auxbasis_response {
                *&mut de_sraux.i_mut((.., atm)).assign(daux_sr.i(p0..p1).sum_axes(0));
            }
        }

        if let Some(factor_j) = self.flags.factor_j {
            de_j *= factor_j;
            de_jaux *= factor_j;
        }
        if let Some(factor_k) = self.flags.factor_k {
            if self.flags.omega.is_some() {
                // rsh case: full exchange = alpha
                let alpha = self.flags.factor_alpha.unwrap_or(0.0);
                de_k *= -alpha;
                de_kaux *= -alpha;
            } else {
                de_k *= -factor_k;
                de_kaux *= -factor_k;
            }
        }
        if self.flags.omega.is_some() {
            // rsh case: short range = - (alpha - hyb)
            let alpha = self.flags.factor_alpha.unwrap_or(0.0);
            let hyb = self.flags.factor_k.unwrap_or(0.0);
            de_sr *= alpha - hyb;
            de_sraux *= alpha - hyb;
        }

        for (key, de_part, flag) in [
            ("de_j", de_j, self.flags.factor_j.is_some()),
            ("de_k", de_k, self.flags.factor_k.is_some()),
            ("de_sr", de_sr, self.flags.omega.is_some()),
            ("de_jaux", de_jaux, self.flags.factor_j.is_some() && self.flags.auxbasis_response),
            ("de_kaux", de_kaux, self.flags.factor_k.is_some() && self.flags.auxbasis_response),
            ("de_sraux", de_sraux, self.flags.omega.is_some() && self.flags.auxbasis_response),
        ] {
            let de_sraw = de_part.into_shape(-1).into_raw();
            let de_part = MatrixFull::from_vec([3, natm], de_sraw).unwrap();
            flag.then(|| self.result.insert(key.into(), de_part));
        }

        return self;
    }

    pub fn calc_de_solvent(&mut self) -> &mut Self {
        if !self.scf_data.mol.ctrl.solvent_enabled {
            let natm = self.scf_data.mol.geom.elem.len();
            self.result.insert("de_solvent".into(), MatrixFull::new([3, natm], 0.0));
            return self;
        }
        let de_solvent = crate::solvent::grad::compute_solvent_gradient(self.scf_data);
        self.result.insert("de_solvent".into(), de_solvent);
        return self;
    }

    pub fn calc(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("uhf grad", "uhf grad");
        time_records.new_item("uhf grad calc_de_nuc", "uhf grad calc_de_nuc");
        time_records.new_item("uhf grad calc_de_ovlp", "uhf grad calc_de_ovlp");
        time_records.new_item("uhf grad calc_de_hcore", "uhf grad calc_de_hcore");
        time_records.new_item("uhf grad calc_de_ext_field", "uhf grad calc_de_ext_field");
        time_records.new_item("uhf grad calc_de_jk", "uhf grad calc_de_jk");
        time_records.new_item("uhf grad calc_de_solvent", "uhf grad calc_de_solvent");
        time_records.new_item("uhf grad calc_de_qmmm", "uhf grad calc_de_qmmm");

        time_records.count_start("uhf grad");

        time_records.count_start("uhf grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("uhf grad calc_de_nuc");

        time_records.count_start("uhf grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("uhf grad calc_de_ovlp");

        time_records.count_start("uhf grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("uhf grad calc_de_hcore");

        if self.flags.ext_field_dipole.is_some() {
            time_records.count_start("uhf grad calc_de_ext_field");
            self.calc_de_ext_field();
            time_records.count("uhf grad calc_de_ext_field");
        }

        time_records.count_start("uhf grad calc_de_qmmm");
        self.calc_de_qmmm();
        time_records.count("uhf grad calc_de_qmmm");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() || self.flags.omega.is_some() {
            time_records.count_start("uhf grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("uhf grad calc_de_jk");
        }

        time_records.count_start("uhf grad calc_de_solvent");
        self.calc_de_solvent();
        time_records.count("uhf grad calc_de_solvent");

        time_records.count("uhf grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_sr").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.get("de_sraux").map(|x| de += x.clone());
        self.result.get("de_qmmm").map(|x| de += x.clone());
        self.result.get("de_solvent").map(|x| de += x.clone());
        self.result.get("de_ext_field").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.scf_data.mol.ctrl.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }
}

fn get_mo_coeff(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 3-D (spin, ao, mo) is better.
    // Currently, vector of 2-D (ao, mo) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let mo_coeff = &scf_data.eigenvectors[spin];
        result.push(rt::asarray((&mo_coeff.data, mo_coeff.size, device)).to_owned());
    }
    return result;
}

fn get_mo_occ(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 2-D (spin, mo) is better.
    // Currently, vector of 1-D (mo) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let mo_occ = &scf_data.occupation[spin];
        result.push(rt::asarray((mo_occ, [mo_occ.len()], device)).to_owned());
    }
    return result;
}

fn get_mo_energy(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 2-D (spin, mo) is better.
    // Currently, vector of 1-D (ao, mo) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let mo_energy = &scf_data.eigenvalues[spin];
        result.push(rt::asarray((mo_energy, [mo_energy.len()], device)).to_owned());
    }
    return result;
}

fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 3-D (spin, ao, ao) is better.
    // Currently, vector of 2-D (ao, ao) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let dm = &scf_data.density_matrix[spin];
        result.push(rt::asarray((&dm.data, dm.size, device)).to_owned());
    }
    return result;
}
