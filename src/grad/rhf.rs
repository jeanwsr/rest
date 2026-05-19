#![warn(unused)]
use crate::grad::traits::GradAPI;
use crate::ri_jk::{self, J2CDecompose};
use crate::scf_io::{self, SCF};
use crate::utilities::memory_batch::*;
use crate::Molecule;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use tensors::MatrixFull;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

#[non_exhaustive]
#[derive(derive_builder::Builder)]
pub struct RIHFGradientFlags {
    /// Print level for debugging.
    #[builder(default = 0)]
    pub print_level: usize,

    /// Memory available for calculation (in MB).
    /// Note that this value does not count the memory used by the program itself.
    #[builder(default = "None")]
    pub max_memory: Option<f64>,

    /// Perform response of auxiliary basis.
    #[builder(default = true)]
    pub auxbasis_response: bool,

    /// Factor of J contribution.
    /// Usually set to 1.0.
    #[builder(default = "Some(1.0)")]
    pub factor_j: Option<f64>,

    /// Factor of K contribution.
    /// For RHF, it should be set to 1.0; for KS, it depends on exchange coefficient.
    #[builder(default = "Some(1.0)")]
    pub factor_k: Option<f64>,
}

/// Gradient structure and values for RHF method.
///
/// Field `result` contains
/// - `de`: total derivative of energy with respect to nuclear coordinates, in unit a.u.
/// - components contributed to total derivative, including `de_nuc`, `de_hcore`, `de_ovlp`, `de_j`,
///   `de_k`, `de_jaux`, `de_kaux`.
pub struct RIRHFGradient<'a> {
    pub scf_data: &'a SCF,
    pub flags: RIHFGradientFlags,
    pub result: HashMap<String, MatrixFull<f64>>,
}

impl GradAPI for RIRHFGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result.get("de").unwrap().clone()
    }

    fn get_energy(&self) -> f64 {
        self.scf_data.scf_energy
    }
}

impl RIRHFGradient<'_> {
    pub fn new(scf_data: &SCF) -> RIRHFGradient<'_> {
        // check SCF type
        match scf_data.scftype {
            scf_io::SCFType::RHF => {},
            _ => panic!("SCFtype is not sutiable for RHF gradient."),
        };

        // // check j2c_decomp flag
        // use crate::ri_jk::decompose::*;
        // match scf_data.mol.ctrl.j2c_decomp.policy {
        //     J2CDecompPolicy::Cd => unimplemented!("Cholesky decompose is not implemented for gradient currently."),
        //     _ => {},
        // };
        // check omega flag
        if scf_data.mol.xc_data.is_rsh() {
            unimplemented!("RI gradient for range-separated hybrid functionals is not implemented currently.")
        }

        // flags
        let mut flags = RIHFGradientFlagsBuilder::default();
        flags.factor_j(Some(1.0));
        flags.factor_k(Some(1.0));
        flags.auxbasis_response(scf_data.mol.ctrl.auxbasis_response);
        flags.print_level(scf_data.mol.ctrl.print_level);
        flags.max_memory(scf_data.mol.ctrl.max_memory);
        let flags = flags.build().unwrap();

        RIRHFGradient { scf_data, flags, result: HashMap::new() }
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

        let dme0 = get_dme0(mo_coeff.view(), mo_occ.view(), mo_energy.view());
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

    pub fn calc_de_jk(&mut self) -> &mut Self {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("de-jk preparation 1", "de-jk preparation 1");
        time_records.new_item("de-jk preparation power", "de-jk preparation power");
        time_records.new_item("de-jk preparation 2", "de-jk preparation 2");
        time_records.new_item("de-jk batch int", "de-jk batch int");
        time_records.new_item("de-jk batch 1", "de-jk batch 1");
        time_records.new_item("de-jk batch 2", "de-jk batch 2");
        time_records.new_item("de-jk batch 3", "de-jk batch 3");
        time_records.new_item("de-jk batch 4", "de-jk batch 4");
        time_records.new_item("de-jk batch 5", "de-jk batch 5");

        time_records.count_start("de-jk preparation 1");

        let mol_obj = &self.scf_data.mol;
        let natm = mol_obj.geom.elem.len();
        let mol = ri_jk::util::get_cint_mol(mol_obj);
        let aux = ri_jk::util::get_cint_aux(mol_obj);
        let device = DeviceBLAS::default();

        // density matrix and triu-packed density matrix
        let dm = get_dm(self.scf_data, &device);
        let dm_tp = pack_triu_tilde(dm.view());

        // orbital coefficients
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let nao = dm.shape()[0];
        let nocc = mo_occ.iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum::<usize>();
        let weighted_occ_coeff = mo_coeff.i((.., 0..nocc)) * mo_occ.mapv(f64::sqrt).i((None, 0..nocc));

        // eigen-decomposed ERI
        let ederi_utp = {
            let tsr = self.scf_data.rimatr.as_ref().unwrap();
            rt::asarray((&tsr.0.data, tsr.0.size, &device))
        };
        let naux = ederi_utp.shape()[0];

        time_records.count_start("de-jk preparation power");
        // tsr_int2c2e_l: J^-1/2
        let j2c_decomp_option = self.scf_data.mol.ctrl.j2c_decomp;
        let j2c_decomp = ri_jk::get_j2c_decomp(&aux, &device, j2c_decomp_option);
        let tsr_int2c2e_l_inv = match j2c_decomp {
            J2CDecompose::Cd { j2c_l, .. } => rt::linalg::inv(j2c_l),
            J2CDecompose::Eig { j2c_l_inv, .. } => j2c_l_inv,
        };
        // TODO: transpose should inside pure functions, try consider uplo
        let tsr_int2c2e_l_inv = tsr_int2c2e_l_inv.into_reverse_axes().into_contig(FlagOrder::F);
        time_records.count("de-jk preparation power");

        // tsr_int2c2e_ip1
        let tsr_int2c2e_ip1 = {
            let (out, shape) = aux.integrate("int2c2e_ip1", "s1", None).into();
            rt::asarray((out, shape, &device))
        };

        // shell partition of int3c2e
        let aux_loc = aux.ao_loc();

        // available memory in MB, if not set, will be calculated from system
        let mem_avail = self.flags.max_memory.map(|max_memory| max_memory - detect_used_memory_mb("proc"));
        let aux_batch_size = calc_batch_size::<f64>(8 * nao * nao, mem_avail, None, Some(naux * nocc * nocc));
        let aux_batch_size = aux_batch_size.min(216);
        let aux_partition = blocksize_partition(&aux_loc, aux_batch_size);

        time_records.count("de-jk preparation 1");

        // preparation finished

        time_records.count_start("de-jk preparation 2");

        // temporaries for de_jaux, de_kaux
        let mut itm_j = rt::full(([], f64::NAN, &device));
        let mut dao_j = rt::full(([], f64::NAN, &device));
        let mut daux_j = rt::full(([], f64::NAN, &device));
        if self.flags.factor_j.is_some() {
            itm_j = get_itm_j(tsr_int2c2e_l_inv.view(), ederi_utp.view(), dm_tp.view());
            dao_j = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
            daux_j = get_grad_daux_j_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_j.view());
        }

        let mut itm_k_occtp = rt::full(([], f64::NAN, &device));
        let mut dao_k = rt::full(([], f64::NAN, &device));
        let mut daux_k = rt::full(([], f64::NAN, &device));
        if self.flags.factor_k.is_some() {
            itm_k_occtp = get_itm_k_occtp(tsr_int2c2e_l_inv.view(), ederi_utp.view(), weighted_occ_coeff.view());
            dao_k = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
            let itm_k_aux = get_itm_k_aux(itm_k_occtp.view_mut());
            daux_k = get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_k_aux.view());
        }

        time_records.count("de-jk preparation 2");

        let mut idx_aux_start = 0;
        for [shl0, shl1] in aux_partition {
            let shl_naux = aux_loc[shl1] - aux_loc[shl0];
            let shl_slices = [[0, mol.nbas()], [0, mol.nbas()], [shl0, shl1]];
            let (p0, p1) = (idx_aux_start, idx_aux_start + shl_naux);

            time_records.count_start("de-jk batch int");
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
            time_records.count("de-jk batch int");

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
                let itm_k_ao = get_itm_k_ao(itm_k_occtp.i((.., p0..p1)), weighted_occ_coeff.view());
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

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        // de_j, de_k, de_jaux, de_kaux
        let mut de_j = rt::full(([3, natm], f64::NAN, &device));
        let mut de_k = rt::full(([3, natm], f64::NAN, &device));
        let mut de_jaux = rt::full(([3, natm], f64::NAN, &device));
        let mut de_kaux = rt::full(([3, natm], f64::NAN, &device));
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

            let [_, _, p0, p1] = aux_slice[atm].clone().try_into().unwrap();
            if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
                *&mut de_jaux.i_mut((.., atm)).assign(daux_j.i(p0..p1).sum_axes(0));
            }
            if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
                *&mut de_kaux.i_mut((.., atm)).assign(daux_k.i(p0..p1).sum_axes(0));
            }
        }

        if let Some(factor_j) = self.flags.factor_j {
            de_j *= factor_j;
            de_jaux *= factor_j;
        }
        if let Some(factor_k) = self.flags.factor_k {
            de_k *= -0.5 * factor_k;
            de_kaux *= -0.5 * factor_k;
        }

        let de_j = {
            let de_j_raw = de_j.into_shape(-1).into_raw();
            MatrixFull::from_vec([3, natm], de_j_raw).unwrap()
        };
        let de_jaux = {
            let de_jaux_raw = de_jaux.into_shape(-1).into_raw();
            MatrixFull::from_vec([3, natm], de_jaux_raw).unwrap()
        };
        let de_k = {
            let de_k_raw = de_k.into_shape(-1).into_raw();
            MatrixFull::from_vec([3, natm], de_k_raw).unwrap()
        };
        let de_kaux = {
            let de_kaux_raw = de_kaux.into_shape(-1).into_raw();
            MatrixFull::from_vec([3, natm], de_kaux_raw).unwrap()
        };

        if self.flags.factor_j.is_some() {
            self.result.insert("de_j".into(), de_j);
        }
        if self.flags.factor_k.is_some() {
            self.result.insert("de_k".into(), de_k);
        }
        if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
            self.result.insert("de_jaux".into(), de_jaux);
        }
        if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
            self.result.insert("de_kaux".into(), de_kaux);
        }

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

    pub fn calc(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("rhf grad", "rhf grad");
        time_records.new_item("rhf grad calc_de_nuc", "rhf grad calc_de_nuc");
        time_records.new_item("rhf grad calc_de_ovlp", "rhf grad calc_de_ovlp");
        time_records.new_item("rhf grad calc_de_hcore", "rhf grad calc_de_hcore");
        time_records.new_item("rhf grad calc_de_jk", "rhf grad calc_de_jk");

        time_records.count_start("rhf grad");

        time_records.count_start("rhf grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("rhf grad calc_de_nuc");

        time_records.count_start("rhf grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("rhf grad calc_de_ovlp");

        time_records.count_start("rhf grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("rhf grad calc_de_hcore");

        time_records.count_start("rhf grad calc_de_qmmm");
        self.calc_de_qmmm();
        time_records.count("rhf grad calc_de_qmmm");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() {
            time_records.count_start("rhf grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("rhf grad calc_de_jk");
        }

        time_records.count("rhf grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.get("de_qmmm").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }
}

pub fn generator_deriv_hcore<'a>(scf_data: &'a SCF) -> impl FnMut(usize) -> Tsr<f64> + 'a {
    let mut mol_obj = scf_data.mol.clone();
    let mol = ri_jk::util::get_cint_mol(&mol_obj);
    let device = DeviceBLAS::default();

    let necp_by_atom = {
        let basis4elem = Molecule::collect_basis(&scf_data.mol.ctrl, &mut scf_data.mol.geom.clone()).0;
        basis4elem.iter().map(|i| if let Some(num_ecp) = i.ecp_electrons { num_ecp } else { 0 }).collect::<Vec<usize>>()
    };
    let has_ecp = necp_by_atom.iter().any(|&x| x > 0);

    let tsr_int1e_ipkin = {
        let (out, shape) = mol.integrate("int1e_ipkin", "s1", None).into();
        rt::asarray((out, shape, &device))
    };

    let tsr_int1e_ipnuc = {
        let (out, shape) = mol.integrate("int1e_ipnuc", "s1", None).into();
        rt::asarray((out, shape, &device))
    };

    let mut h1 = -(tsr_int1e_ipkin + tsr_int1e_ipnuc);

    if has_ecp {
        let (out, shape) = mol.integrate("ECPscalar_ipnuc", "s1", None).into();
        let tsr_int1e_ecp_ipnuc = rt::asarray((out, shape, &device));
        h1 -= tsr_int1e_ecp_ipnuc;
    }
    let aoslice_by_atom = mol_obj.aoslice_by_atom();
    let charge_by_atom = crate::geom_io::get_charge(&mol_obj.geom.elem);

    move |atm_id| {
        let [_, _, p0, p1] = aoslice_by_atom[atm_id];
        mol_obj.with_rinv_at_nucleus(atm_id, |mol_obj| {
            let mol = ri_jk::util::get_cint_mol(&mol_obj);

            let tsr_int1e_iprinv = {
                let (out, shape) = mol.integrate("int1e_iprinv", "s1", None).into();
                rt::asarray((out, shape, &device))
            };

            let mut vrinv = -((&charge_by_atom)[atm_id] - (&necp_by_atom)[atm_id] as f64) * tsr_int1e_iprinv;

            if has_ecp && necp_by_atom[atm_id] > 0 {
                let (out, shape) = mol.integrate("ECPscalar_iprinv", "s1", None).into();
                let tsr_int1e_ecp_iprinv = rt::asarray((out, shape, &device));
                vrinv += tsr_int1e_ecp_iprinv;
            }

            *&mut vrinv.i_mut(p0..p1) += &h1.i(p0..p1);
            (&vrinv + vrinv.swapaxes(0, 1)).into_contig(FlagOrder::F)
        })
    }
}

/* #region utilities */

fn get_mo_coeff(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let mo_coeff = &scf_data.eigenvectors[0];
    return rt::asarray((&mo_coeff.data, mo_coeff.size, device)).to_owned();
}

fn get_mo_occ(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let mo_occ = &scf_data.occupation[0];
    return rt::asarray((mo_occ, [mo_occ.len()], device)).to_owned();
}

fn get_mo_energy(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let mo_energy = &scf_data.eigenvalues[0];
    return rt::asarray((mo_energy, [mo_energy.len()], device)).to_owned();
}

fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let dm = &scf_data.density_matrix[0];
    return rt::asarray((&dm.data, dm.size, device)).to_owned();
}

/* #endregion */

/* #region Matrix Algorithms in RI-RHF code */

pub fn calc_de_nuc(mol: &Molecule) -> MatrixFull<f64> {
    let device = DeviceBLAS::default();

    let natm = mol.geom.elem.len();
    let coords = (0..natm).map(|i| mol.geom.get_coord(i)).flatten().collect::<Vec<f64>>();
    let coords = rt::asarray((&coords, [3, natm], &device));

    let charges_by_atom = crate::geom_io::get_charge(&mol.geom.elem);
    let necp_by_atom = mol
        .basis4elem
        .iter()
        .map(|i| if let Some(num_ecp) = i.ecp_electrons { num_ecp as f64 } else { 0.0 })
        .collect::<Vec<f64>>();
    let charges = rt::asarray((charges_by_atom, &device)) - rt::asarray((necp_by_atom, &device));

    let nuc_z = charges;
    let nuc_v = coords.i((.., None, ..)) - coords.i((.., .., None));
    let nuc_inf = rt::full(([natm], f64::INFINITY, &device));
    let nuc_rinv: Tsr<f64> = 1.0 / (nuc_v.l2_norm_axes(0) + rt::diag(&nuc_inf));

    let tmp =
        -nuc_z.i((None, .., None)) * nuc_z.i((None, None, ..)) * nuc_rinv.mapv(|x| x.powi(3)).i((None, .., ..)) * nuc_v;
    let de_nuc = tmp.sum_axes(1);

    let de_nuc = {
        let de_nuc_raw = de_nuc.into_shape(-1).into_raw();
        MatrixFull::from_vec([3, natm], de_nuc_raw).unwrap()
    };

    return de_nuc;
}

pub fn pack_triu_tilde(dm: TsrView<f64>) -> Tsr<f64> {
    // Pack the lower triangular part of a matrix into a 1D array
    // and non-diagonal values are multiplied by 2.
    assert_eq!(dm.ndim(), 2);
    assert_eq!(dm.shape()[0], dm.shape()[1]);
    let nao = dm.shape()[0];
    let mut dm_triu: Tsr<f64> = 2.0 * dm.pack_triu();
    for i in 0..nao {
        dm_triu[[(i + 2) * (i + 1) / 2 - 1]] *= 0.5;
    }
    return dm_triu;
}

pub fn get_dme0(mo_coeff: TsrView<f64>, mo_occ: TsrView<f64>, mo_energy: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    return (&mo_coeff * mo_occ.i((None, ..)) * mo_energy.i((None, ..))) % mo_coeff.t();
}

pub fn get_grad_dao_ovlp(tsr_int1e_ipovlp: TsrView<f64>, dme0: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int1e_ipovlp.f_prefer());
    assert!(dme0.f_prefer());
    return 2.0 * (tsr_int1e_ipovlp * dme0.i((.., .., None))).sum_axes(1);
}

pub fn get_itm_j(tsr_int2c2e_l_inv: TsrView<f64>, ederi_utp: TsrView<f64>, dm_tp: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    return dm_tp % ederi_utp % tsr_int2c2e_l_inv;
}

pub fn get_grad_daux_j_int2c2e_ip1(tsr_int2c2e_ip1: TsrView<f64>, itm_j: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    let naux = itm_j.shape()[0];
    let device = tsr_int2c2e_ip1.device().clone();
    let mut daux_j_int2c2e_ip1 = rt::zeros(([naux, 3], &device));
    for t in 0..3 {
        *&mut daux_j_int2c2e_ip1.i_mut((.., t)) += &itm_j * (tsr_int2c2e_ip1.i((.., .., t)) % &itm_j);
    }
    return daux_j_int2c2e_ip1;
}

pub fn get_grad_dao_j_int3c2e_ip1(tsr_int3c2e_ip1: TsrView<f64>, dm: TsrView<f64>, itm_j: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip1.f_prefer());

    let nao = dm.shape()[0];
    let naux = itm_j.shape()[0];
    let device = dm.device().clone();

    let mut dao_j_int3c2e_ip1: Tsr<f64> = rt::zeros(([nao, 3], &device));
    for t in 0..3 {
        let tmp1 = tsr_int3c2e_ip1.i((.., .., .., t)).reshape([nao * nao, naux]) % &itm_j;
        *&mut dao_j_int3c2e_ip1.i_mut((.., t)) += (&tmp1.reshape([nao, nao]) * &dm).sum_axes(1);
    }
    dao_j_int3c2e_ip1 *= -2.0;
    return dao_j_int3c2e_ip1;
}

pub fn get_grad_daux_j_int3c2e_ip2(
    tsr_int3c2e_ip2: TsrView<f64>,
    dm_tp: TsrView<f64>,
    itm_j: TsrView<f64>,
) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip2.f_prefer());

    let naux = itm_j.shape()[0];
    let tmp1 = dm_tp % tsr_int3c2e_ip2.reshape((-1, naux * 3));
    return -1.0 * (&tmp1.reshape((naux, 3)) * itm_j.i((.., None)));
}

pub fn get_itm_k_occtp(
    tsr_int2c2e_l_inv: TsrView<f64>,
    ederi_utp: TsrView<f64>,
    weighted_occ_coeff: TsrView<f64>,
) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int2c2e_l_inv.f_prefer());
    assert!(ederi_utp.f_prefer());
    assert!(weighted_occ_coeff.f_prefer());

    let nocc = weighted_occ_coeff.shape()[1];
    let naux = ederi_utp.shape()[1];
    let nocc_tp = nocc * (nocc + 1) / 2;
    let device = tsr_int2c2e_l_inv.device().clone();
    let tmp: Tsr<f64> = unsafe { rt::empty(([nocc_tp, naux], &device)) };
    (0..naux).into_par_iter().for_each(|p| {
        let ederi_bb = ederi_utp.i((.., p)).unpack_triu(FlagSymm::Sy);
        let ederi_oo = weighted_occ_coeff.t() % ederi_bb % &weighted_occ_coeff;

        let mut tmp = unsafe { tmp.force_mut() };
        tmp.i_mut((.., p)).assign(ederi_oo.pack_triu());
    });
    let itm_k_occ = tmp % tsr_int2c2e_l_inv;
    return itm_k_occ;
}

pub fn get_itm_k_aux(mut itm_k_occtp: TsrMut<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(itm_k_occtp.f_prefer());

    let nocc_tp = itm_k_occtp.shape()[0];
    let nocc = ((2 * nocc_tp) as f64).sqrt().floor().to_usize().unwrap();
    assert_eq!(nocc * (nocc + 1) / 2, nocc_tp);

    // modify diag elements in-place
    for i in 0..nocc {
        let idx = (i + 2) * (i + 1) / 2 - 1;
        *&mut itm_k_occtp.i_mut(idx) *= f64::sqrt(0.5);
    }

    let itm_k_aux = 2.0 * (itm_k_occtp.t() % &itm_k_occtp);

    // modify back diag elements in-place
    for i in 0..nocc {
        let idx = (i + 2) * (i + 1) / 2 - 1;
        *&mut itm_k_occtp.i_mut(idx) *= f64::sqrt(2.0);
    }

    return itm_k_aux;
}

pub fn get_itm_k_ao(itm_k_occtp: TsrView<f64>, weighted_occ_coeff: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(itm_k_occtp.f_prefer());
    assert!(weighted_occ_coeff.f_prefer());

    let nao = weighted_occ_coeff.shape()[0];
    let naux = itm_k_occtp.shape()[1];
    let device = weighted_occ_coeff.device().clone();

    let itm_k_ao = unsafe { rt::empty(([nao, nao, naux], &device)) };
    (0..naux).into_par_iter().for_each(|p| {
        let mut itm_k_ao = unsafe { itm_k_ao.force_mut() };
        let itm_k_occ_p = itm_k_occtp.i((.., p)).unpack_triu(FlagSymm::Sy);
        itm_k_ao.i_mut((.., .., p)).assign(&weighted_occ_coeff % itm_k_occ_p % weighted_occ_coeff.t());
    });
    return itm_k_ao;
}

pub fn get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1: TsrView<f64>, itm_k_aux: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int2c2e_ip1.f_prefer());
    assert!(itm_k_aux.f_prefer());

    return (tsr_int2c2e_ip1 * itm_k_aux.i((.., .., None))).sum_axes(1);
}

pub fn get_grad_dao_k_int3c2e_ip1(tsr_int3c2e_ip1: TsrView<f64>, itm_k_ao: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip1.f_prefer());
    assert!(itm_k_ao.f_prefer());

    let naux = itm_k_ao.shape()[2];
    let nao = tsr_int3c2e_ip1.shape()[0];
    let device = tsr_int3c2e_ip1.device().clone();

    let tmp = unsafe { rt::empty(([nao, 3, naux], &device)) };
    (0..naux).into_par_iter().for_each(|p| {
        let mut tmp = unsafe { tmp.force_mut() };
        for t in 0..3 {
            tmp.i_mut((.., t, p))
                .assign(-2.0 * (tsr_int3c2e_ip1.i((.., .., p, t)) * itm_k_ao.i((.., .., p))).sum_axes(1));
        }
    });
    return tmp.sum_axes(-1);
}

pub fn get_grad_daux_k_int3c2e_ip2(tsr_int3c2e_ip2: TsrView<f64>, itm_k_ao: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip2.f_prefer());
    assert!(itm_k_ao.f_prefer());

    let naux = itm_k_ao.shape()[2];
    let nao = itm_k_ao.shape()[0];
    let device = tsr_int3c2e_ip2.device().clone();

    // modify diag elements in-place
    let daux_k_int3c2e_ip2 = rt::zeros(([naux, 3], &device));
    (0..naux).into_par_iter().for_each(|p| {
        let mut itm_k_ao_p = itm_k_ao.i((.., .., p)).pack_triu();
        for u in 0..nao {
            let idx = (u + 2) * (u + 1) / 2 - 1;
            itm_k_ao_p[[idx]] *= 0.5;
        }
        let tmp = -2.0 * (itm_k_ao_p % tsr_int3c2e_ip2.i((.., p)));

        let mut daux_k_int3c2e_ip2 = unsafe { daux_k_int3c2e_ip2.force_mut() };
        *&mut daux_k_int3c2e_ip2.i_mut(p) += tmp;
    });
    return daux_k_int3c2e_ip2;
}

/* #endregion */
