//! Optimized RI-JK Hessian computation.
//!
//! Algorithm is somehow optimized, and of no reference in current state.
//! Author of first implementation: Andrew J. Zhu <ajz34@outlook.com>
//!
//! The optimization route is not following any article, or codebase that I (ajz34) know, and is not
//! following AI assistance. If there is coincidence, it is accidental.
//!
//! The correct value is compared to PySCF (written by Qiming Sun). Reference article:
//! > Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models
//! > Chen, et al. arXiv:1906.09427
//!
//! PySCF code referenced the ORCA's implementation:
//! > Efficient implementation of the analytic second derivatives of Hartree-Fock and hybrid DFT
//! > energies: a detailed analysis of different approximations
//! > Bykov, et al. Mol Phys. 113, 1961 (2015). DOI: 10.1080/00268976.2015.1025114

use super::prelude_dev::*;
use crate::grad::rhf::pack_triu_tilde;
use crate::analdrv::hessian::cint_handling::*;
use crate::analdrv::prelude::*;
use crate::ri_jk::util::*;

use crate::ri_jk::decompose::*;
use crate::ri_jk::pure_decompose::{get_j2c_decomp, solve_by_j2c, solve_by_j2c_mut};

/* #region skeleton derivative keys */

pub const KEYS_J20: [&str; 3] = ["de_J20_1", "de_J20_2", "de_J20_3"];
pub const KEYS_K20: [&str; 4] = ["de_K20_1a", "de_K20_1b", "de_K20_2", "de_K20_3"];
pub const KEYS_J11: [&str; 4] = ["de_J11_1", "de_J11_2", "de_J11_3", "de_J11_4"];
pub const KEYS_K11: [&str; 4] = ["de_K11_1", "de_K11_2", "de_K11_3", "de_K11_4"];
pub const KEYS_J02: [&str; 9] =
    ["de_J02_1", "de_J02_2", "de_J02_3a", "de_J02_3b", "de_J02_4", "de_J02_5", "de_J02_6", "de_J02_7", "de_J02_8"];
pub const KEYS_K02: [&str; 9] =
    ["de_K02_1", "de_K02_2", "de_K02_3a", "de_K02_3b", "de_K02_4", "de_K02_5", "de_K02_6", "de_K02_7", "de_K02_8"];

pub const KEYS_J1AO: [&str; 5] = ["j1ao_aux0", "j1ao_aux1_1", "j1ao_aux1_2", "j1ao_aux1_3", "j1ao_aux1_4"];
pub const KEYS_K1BRA: [&str; 8] = [
    "k1bra_aux0_1",
    "k1bra_aux0_2",
    "k1bra_aux0_3",
    "k1bra_aux0_4",
    "k1bra_aux1_1",
    "k1bra_aux1_2",
    "k1bra_aux1_3",
    "k1bra_aux1_4",
];

/* #endregion */

/* #region skeleton */

macro_rules! tic {
    ($timing:expr, $t0:expr, $msg:expr) => {
        let t1 = std::time::Instant::now();
        let dt = t1.duration_since($t0).as_secs_f64();
        $timing.push(($msg.to_string(), dt));
    };
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn get_rijk_skeleton_decomposed_separated(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView],
    mo_occ: &[TsrView],
    cderi: TsrView,
    j2c_decomp: &J2CDecompose,
    do_j: bool,
    do_k: bool,
    nbatch_aux: usize,
    atm_list: Option<&[usize]>,
    dm0: Option<TsrView>,
) -> (Option<HashMap<&'static str, Tsr>>, Vec<HashMap<&'static str, Tsr>>, Vec<(String, f64)>) {
    let device = cderi.device().clone();
    let mut timing = vec![];
    let time_full = std::time::Instant::now();

    // --- basic checks --- //
    if mo_coeff.is_empty() || mo_occ.is_empty() {
        panic!("mo_coeff and mo_occ must be non-empty");
    }
    if mo_coeff.len() != mo_occ.len() {
        panic!("mo_coeff and mo_occ must have the same length");
    }
    let nset = mo_coeff.len();

    // --- prepare shared --- //
    let t0 = std::time::Instant::now();
    let (dims, aoslices, auxslices, aux_ranges, shared, solve_aux) =
        prepare_shared(mol, aux, j2c_decomp, nbatch_aux, atm_list, &device);
    tic!(timing, t0, "prepare_shared");

    // --- prepare j --- //
    let j_in = do_j.then(|| {
        let t0 = std::time::Instant::now();
        // - dm0: [nao, nao]; note this density is total density instead of spin-separated density
        let dm0 = dm0.map_or_else(
            || {
                let nao = dims["nao"];
                let mut dm0 = rt::zeros(([nao, nao], &device));
                for iset in 0..nset {
                    dm0 += get_dm0_restricted(mo_coeff[iset].view(), mo_occ[iset].view())
                }
                dm0
            },
            |dm0| dm0.to_owned(),
        );
        let j_in = prepare_j(&solve_aux, &dims, dm0.view(), cderi.view());
        tic!(timing, t0, "prepare_j");
        j_in
    });
    let mut j_out = do_j.then(HashMap::new);
    let mut j_intmd = do_j.then(HashMap::new);

    // --- prepare k --- //

    let mut k_ins = vec![];
    if do_k {
        for iset in 0..nset {
            let t0 = std::time::Instant::now();
            let k_in = prepare_k(&solve_aux, &dims, mo_coeff[iset].view(), mo_occ[iset].view(), cderi.view());
            k_ins.push(k_in);
            tic!(timing, t0, &format!("prepare_k {iset}"));
        }
    }
    let mut k_outs = (0..k_ins.len()).map(|_| HashMap::new()).collect_vec();
    let mut k_intmds = (0..k_ins.len()).map(|_| HashMap::new()).collect_vec();

    // --- evaluate oneshot --- //

    let t0 = std::time::Instant::now();
    let timing_oneshot = evaluate_oneshot(
        &dims,
        mol,
        aux,
        &aoslices,
        &auxslices,
        &aux_ranges,
        &device,
        j_in.as_ref(),
        &k_ins,
        j_out.as_mut(),
        &mut k_outs,
    );
    timing.extend(timing_oneshot);
    tic!(timing, t0, "evaluate_oneshot");

    // --- evaluate j2c-derivatives-only terms --- //

    let t0 = std::time::Instant::now();
    let timing_j2c_deriv_only = evaluate_j2c_deriv_only(
        &dims,
        &shared,
        aux,
        &auxslices,
        &device,
        j_in.as_ref(),
        &k_ins,
        j_out.as_mut(),
        &mut k_outs,
    );
    timing.extend(timing_j2c_deriv_only);
    tic!(timing, t0, "evaluate_j2c_deriv_only");

    // --- evaluate jk1 j2c-skeleton terms --- //

    let t0 = std::time::Instant::now();
    let timing_jk1_j2c_deriv = evaluate_jk1_j2c_deriv(
        &dims,
        &shared,
        cderi.view(),
        &auxslices,
        &device,
        &solve_aux,
        j_in.as_ref(),
        &k_ins,
        j_out.as_mut(),
        &mut k_outs,
    );
    timing.extend(timing_jk1_j2c_deriv);
    tic!(timing, t0, "evaluate_jk1_j2c_deriv");

    // --- evaluate j3c-ip2 related terms --- //

    let t0 = std::time::Instant::now();
    let timing_j3c_ip2 = evaluate_j3c_ip2(
        &dims,
        &shared,
        cderi.view(),
        mol,
        aux,
        &auxslices,
        &aux_ranges,
        &device,
        &solve_aux,
        j_in.as_ref(),
        &mut k_ins,
        j_out.as_mut(),
        &mut k_outs,
        j_intmd.as_mut(),
        &mut k_intmds,
    );
    timing.extend(timing_j3c_ip2);
    tic!(timing, t0, "evaluate_j3c_ip2");

    // --- evaluate j3c-ip1 related terms --- //

    let t0 = std::time::Instant::now();
    let timing_j3c_ip1 = evaluate_j3c_ip1(
        &dims,
        &shared,
        cderi.view(),
        mol,
        aux,
        &aoslices,
        &auxslices,
        &aux_ranges,
        &device,
        &solve_aux,
        j_in.as_ref(),
        &mut k_ins,
        j_out.as_mut(),
        &mut k_outs,
        j_intmd.as_mut(),
        &mut k_intmds,
    );
    timing.extend(timing_j3c_ip1);
    tic!(timing, t0, "evaluate_j3c_ip1");

    tic!(timing, time_full, "get_rijk_skeleton_decomposed_separated");
    (j_out, k_outs, timing)
}

pub type FnSolveAux<'a> = Box<dyn Fn(TsrMut, FlagSide, bool) + 'a>;
type PrepareSharedOutput<'a> = (
    HashMap<&'static str, usize>, // dims
    Vec<[usize; 4]>,              // aoslices
    Vec<[usize; 4]>,              // auxslices
    Vec<[usize; 4]>,              // aux_ranges
    HashMap<&'static str, Tsr>,   // shared (intermediates)
    FnSolveAux<'a>,               // solve_aux(tsr_mut, left/right, do_flip)
);

pub fn prepare_shared<'a>(
    mol: &CInt,
    aux: &CInt,
    j2c_decomp: &'a J2CDecompose,
    nbatch_aux: usize,
    atm_list: Option<&[usize]>,
    device: &DeviceBLAS,
) -> PrepareSharedOutput<'a> {
    // aoslices, auxslices
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |list| list.to_vec());
    let natm = atm_list.len();
    let aoslices = mol.aoslice_by_atom();
    let auxslices = aux.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&i| aoslices[i]).collect_vec();
    let auxslices = atm_list.iter().map(|&i| auxslices[i]).collect_vec();

    // aux_ranges
    let aux_balance = aux.balance_partition(nbatch_aux);
    let mut p0 = 0;
    let aux_ranges = aux_balance
        .into_iter()
        .map(|[sh0, sh1, size]| {
            let range = [sh0, sh1, p0, p0 + size];
            p0 += size;
            range
        })
        .collect_vec();

    let solve_aux =
        |tsr_mut: TsrMut, side: FlagSide, do_flip: bool| solve_by_j2c_mut(tsr_mut, j2c_decomp, side, do_flip);

    let j2c_ip1 = hess_intor(aux, "int2c2e_ip1", "s1", None, device);
    let mut rcd_j2c_ip1 = j2c_ip1.to_owned();
    rcd_j2c_ip1.axes_iter_mut(-1).for_each(|m| solve_aux(m, Right, false));
    let mut rrcd_j2c_ip1 = rcd_j2c_ip1.to_owned();
    rrcd_j2c_ip1.axes_iter_mut(-1).for_each(|m| solve_aux(m, Right, true));

    let naux = aux.nao();
    let j2c_inv = {
        let mut eye = rt::eye((naux, device));
        solve_aux(eye.view_mut(), Right, true);
        let mut out = eye.t().into_contig(ColMajor);
        solve_aux(out.view_mut(), Right, true);
        out
    };

    let dims = HashMap::from([("natm", natm), ("nao", mol.nao()), ("naux", aux.nao())]);
    let shared = HashMap::from([
        ("j2c_ip1", j2c_ip1),
        ("rcd_j2c_ip1", rcd_j2c_ip1),
        ("rrcd_j2c_ip1", rrcd_j2c_ip1),
        ("j2c_inv", j2c_inv),
    ]);

    (dims, aoslices, auxslices, aux_ranges, shared, Box::new(solve_aux))
}

pub fn prepare_j(
    solve_aux: &FnSolveAux,
    dims: &HashMap<&'static str, usize>,
    dm0: TsrView,
    cderi: TsrView,
) -> HashMap<&'static str, Tsr> {
    let nao = dims["nao"];
    assert_eq!(dm0.shape(), &[nao, nao], "dm0 in prepare_j must be a single square matrix of shape [nao, nao]");

    // pack density matrix with scaled (off-diag, diag)
    let dm0_tp = pack_triu_tilde(dm0.view());
    let rrcd_eri_aux = {
        let mut out = dm0_tp.view() % cderi;
        solve_aux(out.view_mut(), Right, true);
        out
    };
    HashMap::from([("dm0", dm0.to_owned()), ("dm0_tp", dm0_tp), ("rrcd_eri_aux", rrcd_eri_aux)])
}

pub fn prepare_k(
    solve_aux: &FnSolveAux,
    dims: &HashMap<&'static str, usize>,
    mo_coeff: TsrView,
    mo_occ: TsrView,
    cderi: TsrView,
) -> HashMap<&'static str, Tsr> {
    let nao = dims["nao"];
    let naux = dims["naux"];
    let nmo = mo_coeff.shape()[1];
    let device = cderi.device().clone();
    assert_eq!(mo_coeff.shape(), &[nao, nmo], "mo_coeff in prepare_k must be of shape [nao, nmo]");
    assert_eq!(mo_occ.shape(), &[nmo], "mo_occ in prepare_k must be of shape [nmo]");

    let occidx = mo_occ.view().greater(0).into_vec();
    let nocc = occidx.iter().filter(|&&x| x).count();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let mocc_2 = &mocc * occ.view().sqrt().i((None, ..));
    let occ_invsqrt = occ.view().pow(-0.5);

    // orbital transformation
    let rcd_eri_bra: Tsr = rt::zeros(([nao, nocc, naux], &device));
    let rcd_eri_occ: Tsr = rt::zeros(([nocc, nocc, naux], &device));
    (0..naux).into_par_iter().for_each(|p| {
        let rcd_eri_bra_p = rcd_eri_bra.i((.., .., p));
        let rcd_eri_occ_p = rcd_eri_occ.i((.., .., p));
        let mut rcd_eri_bra_p = unsafe { rcd_eri_bra_p.force_mut() };
        let mut rcd_eri_occ_p = unsafe { rcd_eri_occ_p.force_mut() };
        let cderi_p = cderi.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
        rcd_eri_bra_p.matmul_from(&cderi_p, &mocc_2, 1.0, 0.0);
        rcd_eri_occ_p.matmul_from(&mocc_2.t(), &rcd_eri_bra_p, 1.0, 0.0);
    });

    // j2c solve (consumes previous results in-place)
    let mut rrcd_eri_bra = rcd_eri_bra;
    solve_aux(rrcd_eri_bra.view_mut(), Right, true);
    let mut rrcd_eri_occ = rcd_eri_occ;
    solve_aux(rrcd_eri_occ.view_mut(), Right, true);

    let fold_eri_bra = &mocc_2 % &rrcd_eri_occ;

    HashMap::from([
        ("mocc", mocc),
        ("mocc_2", mocc_2),
        ("occ_invsqrt", occ_invsqrt),
        ("rrcd_eri_bra", rrcd_eri_bra),
        ("rrcd_eri_occ", rrcd_eri_occ),
        ("fold_eri_bra", fold_eri_bra),
    ])
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_oneshot(
    dims: &HashMap<&'static str, usize>,
    mol: &CInt,
    aux: &CInt,
    aoslices: &[[usize; 4]],
    auxslices: &[[usize; 4]],
    aux_ranges: &[[usize; 4]],
    device: &DeviceBLAS,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &[HashMap<&'static str, Tsr>],
    j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let nao = dims["nao"];
    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();

    // --- integral generators --- //

    let gen_j3c_ipvip1 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ipvip1", "s1", device);
    let gen_j3c_ipip1 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ipip1", "s1", device);
    let gen_j3c_ip1ip2 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ip1ip2", "s1", device);
    let gen_j3c_ipip2 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ipip2", "s1", device);

    // --- dbas allocations --- //

    let mut dbas_j: HashMap<&str, Tsr> = HashMap::new();
    let mut dbas_ks: Vec<HashMap<&str, Tsr>> = (0..nset_k).map(|_| HashMap::new()).collect_vec();
    if do_j {
        dbas_j.insert("J20_2", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_j.insert("J20_3", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_j.insert("J11_1", rt::zeros(([nao, naux, 3, 3], device)));
        dbas_j.insert("J02_1", rt::zeros(([naux, 3, 3], device)));
    }
    for i in 0..nset_k {
        let dbas_k = &mut dbas_ks[i];
        dbas_k.insert("K20_2", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_k.insert("K20_3", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_k.insert("K11_1", rt::zeros(([nao, naux, 3, 3], device)));
        dbas_k.insert("K02_1", rt::zeros(([naux, 3, 3], device)));
    }

    // --- dbas evaluation --- //

    for &[sh0, sh1, p0, p1] in aux_ranges {
        // --- common tensors --- //

        let t0 = std::time::Instant::now();

        // j-part
        let rrcd_eri_aux = j_in.map(|j_in| j_in["rrcd_eri_aux"].i(p0..p1));

        // k-part
        let mut tmps_k_ao = vec![];
        for iset in 0..nset_k {
            let k_in = &k_ins[iset];
            let mocc_2 = &k_in["mocc_2"];
            let rrcd_eri_occ = k_in["rrcd_eri_occ"].i((.., .., p0..p1));
            tmps_k_ao.push(mocc_2 % rrcd_eri_occ % mocc_2.t());
        }

        tic!(timing, t0, &format!("evaluate_oneshot, common aux({p0}:{p1})"));

        // --- 20-2 (ipvip1) --- //

        let t0 = std::time::Instant::now();
        let j3c_ipvip1 = gen_j3c_ipvip1([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ipvip1 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ipvip1, rrcd_eri_aux.i((None, None, ..)), 2);
            *dbas_j.get_mut("J20_2").unwrap() += tmp1 * dm0;
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ipvip1, tmp_k_ao, 2);
            *dbas_ks[iset].get_mut("K20_2").unwrap() += tmp1;
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 20-2 aux({p0}:{p1})"));
        drop(j3c_ipvip1);

        // --- 20-3 (ipip1) --- //

        let t0 = std::time::Instant::now();
        let j3c_ipip1 = gen_j3c_ipip1([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ipip1 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ipip1, rrcd_eri_aux.i((None, None, ..)), 2);
            *dbas_j.get_mut("J20_3").unwrap() += tmp1 * dm0;
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ipip1, tmp_k_ao, 2);
            *dbas_ks[iset].get_mut("K20_3").unwrap() += tmp1;
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 20-3 aux({p0}:{p1})"));
        drop(j3c_ipip1);

        // --- 11-1 (ip1ip2) --- //

        let t0 = std::time::Instant::now();
        let j3c_ip1ip2 = gen_j3c_ip1ip2([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ip1ip2 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ip1ip2, &dm0, 1) * rrcd_eri_aux.i((None, ..));
            dbas_j.get_mut("J11_1").unwrap().i_mut((.., p0..p1)).assign(tmp1);
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ip1ip2, tmp_k_ao, 1);
            dbas_ks[iset].get_mut("K11_1").unwrap().i_mut((.., p0..p1)).assign(tmp1);
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 11-1 aux({p0}:{p1})"));
        drop(j3c_ip1ip2);

        // --- 02-1 (ipip2) --- //

        let t0 = std::time::Instant::now();
        let j3c_ipip2 = gen_j3c_ipip2([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ipip2 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ipip2, &dm0, ([0, 1], [0, 1])) * rrcd_eri_aux;
            dbas_j.get_mut("J02_1").unwrap().i_mut(p0..p1).assign(tmp1);
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ipip2, tmp_k_ao, ([0, 1], [0, 1]));
            dbas_ks[iset].get_mut("K02_1").unwrap().i_mut(p0..p1).assign(tmp1);
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 02-1 aux({p0}:{p1})"));
        drop(j3c_ipip2);
    }

    // --- reduce to hessian contribution --- //

    let t0 = std::time::Instant::now();

    if do_j {
        let j_out = j_out.unwrap();

        let mut de_J20_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J20_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J11_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_1: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_j["J20_3"].i(slcA).sum_axes([0, 1]);
            de_J20_3.i_mut((.., .., A, A)).assign(tmp);

            for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_j["J20_2"].i((slcA, slcB)).sum_axes([0, 1]);
                de_J20_2.i_mut((.., .., B, A)).assign(tmp);
            }

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_j["J11_1"].i((slcA, slcB)).sum_axes([0, 1]);
                de_J11_1.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            let tmp = dbas_j["J02_1"].i(slcA).sum_axes(0);
            de_J02_1.i_mut((.., .., A, A)).assign(tmp);
        }

        let scale_J20_2 = 1.0;
        let scale_J20_3 = 1.0;
        let scale_J11_1 = 2.0;
        let scale_J02_1 = 0.5;
        j_out.insert("de_J20_2", scale_J20_2 * (&de_J20_2 + &de_J20_2.transpose([1, 0, 3, 2])));
        j_out.insert("de_J20_3", scale_J20_3 * (&de_J20_3 + &de_J20_3.transpose([1, 0, 3, 2])));
        j_out.insert("de_J11_1", scale_J11_1 * (&de_J11_1 + &de_J11_1.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_1", scale_J02_1 * (&de_J02_1 + &de_J02_1.transpose([1, 0, 3, 2])));
    }

    for iset in 0..nset_k {
        let k_out = &mut k_outs[iset];

        let mut de_K20_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K20_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K11_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_1: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_ks[iset]["K20_3"].i(slcA).sum_axes([0, 1]);
            de_K20_3.i_mut((.., .., A, A)).assign(tmp);

            for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_ks[iset]["K20_2"].i((slcA, slcB)).sum_axes([0, 1]);
                de_K20_2.i_mut((.., .., B, A)).assign(tmp);
            }

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_ks[iset]["K11_1"].i((slcA, slcB)).sum_axes([0, 1]);
                de_K11_1.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            let tmp = dbas_ks[iset]["K02_1"].i(slcA).sum_axes(0);
            de_K02_1.i_mut((.., .., A, A)).assign(tmp);
        }

        let scale_K20_2 = 1.0;
        let scale_K20_3 = 1.0;
        let scale_K11_1 = 2.0;
        let scale_K02_1 = 0.5;
        k_out.insert("de_K20_2", scale_K20_2 * (&de_K20_2 + &de_K20_2.transpose([1, 0, 3, 2])));
        k_out.insert("de_K20_3", scale_K20_3 * (&de_K20_3 + &de_K20_3.transpose([1, 0, 3, 2])));
        k_out.insert("de_K11_1", scale_K11_1 * (&de_K11_1 + &de_K11_1.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_1", scale_K02_1 * (&de_K02_1 + &de_K02_1.transpose([1, 0, 3, 2])));
    }

    tic!(timing, t0, "evaluate_oneshot, reduce to hessian");

    timing
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_j2c_deriv_only(
    dims: &HashMap<&'static str, usize>,
    shared: &HashMap<&'static str, Tsr>,
    aux: &CInt,
    auxslices: &[[usize; 4]],
    device: &DeviceBLAS,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &[HashMap<&'static str, Tsr>],
    j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();

    let j2c_ip1 = shared["j2c_ip1"].view();
    let rcd_j2c_ip1 = shared["rcd_j2c_ip1"].view();
    let rrcd_j2c_ip1 = shared["rrcd_j2c_ip1"].view();
    let j2c_inv = shared["j2c_inv"].view();

    // --- integral evaluation --- //

    let t0 = std::time::Instant::now();
    let j2c_ipip1 = hess_intor(aux, "int2c2e_ipip1", "s1", None, device);
    let j2c_ip1ip2 = hess_intor(aux, "int2c2e_ip1ip2", "s1", None, device);
    tic!(timing, t0, "evaluate_j2c_deriv_only, integration");

    // --- evaluation j-part --- //

    if do_j {
        let t0 = std::time::Instant::now();

        let j_out = j_out.unwrap();
        let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();

        // --- dbas evaluation --- //

        let dbas_J02_2 = rt::vecdot(&j2c_ipip1, &rrcd_eri_aux, 0) * &rrcd_eri_aux;
        let dbas_J02_3a = &j2c_ip1ip2 * &rrcd_eri_aux * rrcd_eri_aux.i((None, ..));
        let tmp1 = &rcd_j2c_ip1 * &rrcd_eri_aux;
        let dbas_J02_3b = tmp1.i((.., .., None, ..)) % tmp1.i((.., .., .., None)).swapaxes(0, 1);
        let tmp1 = rt::vecdot(&j2c_ip1, &rrcd_eri_aux, 0);
        let dbas_J02_6 = tmp1.i((.., None, None, ..)) * &j2c_inv * tmp1.i((None, .., .., None));
        let tmp2 = &rrcd_j2c_ip1 * &rrcd_eri_aux;
        let dbas_J02_8 = tmp1.i((None, .., .., None)) * tmp2.i((.., .., None, ..));

        // --- reduce to hessian contribution --- //

        let mut de_J02_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_3a: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_3b: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_6: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_8: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_J02_2.i(slcA).sum_axes(0);
            de_J02_2.i_mut((.., .., A, A)).assign(tmp);

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_J02_3a.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_3a.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J02_3b.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_3b.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J02_6.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_6.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J02_8.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_8.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_J02_2 = -0.5;
        let scale_J02_3a = -0.5;
        let scale_J02_3b = 0.5;
        let scale_J02_6 = 0.5;
        let scale_J02_8 = -1;
        j_out.insert("de_J02_2", scale_J02_2 * (&de_J02_2 + &de_J02_2.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_3a", scale_J02_3a * (&de_J02_3a + &de_J02_3a.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_3b", scale_J02_3b * (&de_J02_3b + &de_J02_3b.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_6", scale_J02_6 * (&de_J02_6 + &de_J02_6.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_8", scale_J02_8 * (&de_J02_8 + &de_J02_8.transpose([1, 0, 3, 2])));

        tic!(timing, t0, "evaluate_j2c_deriv_only, evaluate j-part");
    }

    // --- evaluation (K) --- //

    for iset in 0..nset_k {
        let t0 = std::time::Instant::now();

        let k_in = &k_ins[iset];
        let k_out = &mut k_outs[iset];
        let rrcd_eri_occ = k_in["rrcd_eri_occ"].view();
        let fold_eri_aux = rrcd_eri_occ.reshape((-1, naux)).t() % rrcd_eri_occ.reshape((-1, naux));

        // --- dbas evaluation --- //

        let dbas_K02_2 = rt::vecdot(&j2c_ipip1, &fold_eri_aux, 0);
        let dbas_K02_3a = &j2c_ip1ip2 * &fold_eri_aux;
        let dbas_K02_3b =
            rcd_j2c_ip1.i((.., .., None, ..)) % rcd_j2c_ip1.i((.., .., .., None)).swapaxes(0, 1) * &fold_eri_aux;
        let dbas_K02_6 = j2c_ip1.i((.., .., None, ..)) % &fold_eri_aux % j2c_ip1.i((.., .., .., None)) * &j2c_inv;
        let dbas_K02_8 = fold_eri_aux % j2c_ip1.i((.., .., .., None)) * rrcd_j2c_ip1.i((.., .., None, ..));

        // --- reduce to hessian contribution --- //

        let mut de_K02_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_3a: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_3b: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_6: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_8: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_K02_2.i(slcA).sum_axes(0);
            de_K02_2.i_mut((.., .., A, A)).assign(tmp);

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_K02_3a.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_3a.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_3b.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_3b.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_6.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_6.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_8.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_8.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_K02_2 = -0.5;
        let scale_K02_3a = -0.5;
        let scale_K02_3b = 0.5;
        let scale_K02_6 = -0.5;
        let scale_K02_8 = -1.0;
        k_out.insert("de_K02_2", scale_K02_2 * (&de_K02_2 + &de_K02_2.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_3a", scale_K02_3a * (&de_K02_3a + &de_K02_3a.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_3b", scale_K02_3b * (&de_K02_3b + &de_K02_3b.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_6", scale_K02_6 * (&de_K02_6 + &de_K02_6.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_8", scale_K02_8 * (&de_K02_8 + &de_K02_8.transpose([1, 0, 3, 2])));

        tic!(timing, t0, &format!("evaluate_j2c_deriv_only, evaluate k-part {iset}"));
    }

    timing
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_jk1_j2c_deriv(
    dims: &HashMap<&'static str, usize>,
    shared: &HashMap<&'static str, Tsr>,
    cderi: TsrView,
    auxslices: &[[usize; 4]],
    device: &DeviceBLAS,
    solve_aux: &FnSolveAux,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &[HashMap<&'static str, Tsr>],
    j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let nao = dims["nao"];
    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();
    let j2c_ip1 = shared["j2c_ip1"].view();
    let rcd_j2c_ip1 = shared["rcd_j2c_ip1"].view();

    // --- evaluation j-part --- //

    if do_j {
        let t0 = std::time::Instant::now();

        let j_out = j_out.unwrap();
        let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();

        // --- j1ao_aux1_3 --- //

        let tmp1 = -rt::vecdot(&j2c_ip1, &rrcd_eri_aux, 0);
        let mut tmp2: Tsr = rt::zeros(([naux, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            tmp2.i_mut((slcA, .., A)).assign(tmp1.i(slcA));
        }
        solve_aux(tmp2.view_mut(), Left, true);
        let j1ao_aux1_3_tp = &cderi % tmp2.reshape((naux, -1));
        let j1ao_aux1_3 = j1ao_aux1_3_tp.reshape((-1, 3, natm)).unpack_tri(Upper, FlagSymm::Sy);
        j_out.insert("j1ao_aux1_3", j1ao_aux1_3);

        // --- j1ao_aux1_4 --- //

        let tmp1 = &rcd_j2c_ip1 * &rrcd_eri_aux;
        let mut tmp2: Tsr = rt::zeros(([naux, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            tmp2.i_mut((.., .., A)).assign(tmp1.i(slcA).sum_axes(0));
        }
        let j1ao_aux1_4_tp = &cderi % tmp2.reshape((naux, -1));
        let j1ao_aux1_4 = j1ao_aux1_4_tp.reshape((-1, 3, natm)).unpack_tri(Upper, FlagSymm::Sy);
        j_out.insert("j1ao_aux1_4", j1ao_aux1_4);

        tic!(timing, t0, "evaluate_jk1_j2c_deriv, j-part");
    }

    // --- evaluation k-part --- //

    for iset in 0..nset_k {
        let t0 = std::time::Instant::now();

        let k_in = &k_ins[iset];
        let k_out = &mut k_outs[iset];
        let rrcd_eri_occ = k_in["rrcd_eri_occ"].view();
        let rrcd_eri_bra = k_in["rrcd_eri_bra"].view();
        let occ_invsqrt = k_in["occ_invsqrt"].view();
        let nocc = occ_invsqrt.shape()[0];

        // --- k1bra_aux1_3 --- //

        let mut k1bra_aux1_3 = rt::zeros(([nao, nocc, 3, natm], device));
        for t in 0..3 {
            let tmp1 = rrcd_eri_bra.reshape([nao * nocc, naux]) % j2c_ip1.i((.., .., t)).t();
            let tmp1 = tmp1.into_shape([nao, nocc, naux]);
            for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
                let slcA = rt::slice!(p0A, p1A);
                let tmp =
                    tmp1.i((.., .., slcA)).reshape((nao, -1)) % rrcd_eri_occ.i((.., .., slcA)).reshape((nocc, -1)).t();
                k1bra_aux1_3.i_mut((.., .., t, A)).assign(tmp);
            }
        }
        k1bra_aux1_3 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux1_3", k1bra_aux1_3);

        // --- k1bra_aux1_4 --- //

        let mut k1bra_aux1_4 = rt::zeros(([nao, nocc, 3, natm], device));
        for t in 0..3 {
            let tmp1 = rrcd_eri_occ.reshape([nocc * nocc, naux]) % j2c_ip1.i((.., .., t)).t();
            let tmp1 = tmp1.into_shape([nocc, nocc, naux]);
            for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
                let slcA = rt::slice!(p0A, p1A);
                let tmp =
                    rrcd_eri_bra.i((.., .., slcA)).reshape((nao, -1)) % tmp1.i((.., .., slcA)).reshape((nocc, -1)).t();
                k1bra_aux1_4.i_mut((.., .., t, A)).assign(tmp);
            }
        }
        k1bra_aux1_4 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux1_4", k1bra_aux1_4);

        tic!(timing, t0, &format!("evaluate_jk1_j2c_deriv, k-part {iset}"));
    }

    timing
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_j3c_ip2(
    dims: &HashMap<&'static str, usize>,
    shared: &HashMap<&'static str, Tsr>,
    cderi: TsrView,
    mol: &CInt,
    aux: &CInt,
    auxslices: &[[usize; 4]],
    aux_ranges: &[[usize; 4]],
    device: &DeviceBLAS,
    solve_aux: &FnSolveAux,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &mut [HashMap<&'static str, Tsr>],
    mut j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
    mut j_intmd: Option<&mut HashMap<&'static str, Tsr>>,
    k_intmds: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let nao = dims["nao"];
    let naux = dims["naux"];
    let natm = dims["natm"];
    let j2c_ip1 = shared["j2c_ip1"].view();
    let rrcd_j2c_ip1 = shared["rrcd_j2c_ip1"].view();
    let j2c_inv = shared["j2c_inv"].view();
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();

    let gen_j3c_ip2 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ip2", "s1", device);

    // --- integral contract --- //

    // for this part, we will evaluate integrals by batch and generate the intermediates; so that in
    // future we will not evaluate these integrals again.

    if let Some(j_intmd) = j_intmd.as_mut() {
        j_intmd.insert("j3c_ip2_aux", rt::zeros(([naux, 3], device)));
        j_intmd.insert("j1ao_aux1_1", rt::zeros(([nao, nao, 3, natm], device)));
    }
    for (k_in, k_intmd) in k_ins.iter_mut().zip(k_intmds.iter_mut()) {
        let nocc = k_in["occ_invsqrt"].shape()[0];
        k_intmd.insert("j3c_ip2_occ", rt::zeros(([nocc, nocc, naux, 3], device)));
        k_intmd.insert("k1bra_aux1_2", rt::zeros(([nao, nocc, 3, natm], device)));
    }

    for &[sh0, sh1, p0, p1] in aux_ranges.iter() {
        let t0 = std::time::Instant::now();
        let j3c_ip2_batch = gen_j3c_ip2([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_j3c_ip2, j3c_ip2 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();

            // --- j3c_ip2_aux --- //

            let mut j3c_ip2_aux = j_intmd.as_mut().unwrap().get_mut("j3c_ip2_aux").unwrap().view_mut();
            let tmp = rt::vecdot(&j3c_ip2_batch, &dm0, ([0, 1], [0, 1]));
            j3c_ip2_aux.i_mut(p0..p1).assign(tmp);

            // --- j1ao_aux1_1 --- //

            let mut j1ao_aux1_1 = j_intmd.as_mut().unwrap().get_mut("j1ao_aux1_1").unwrap().view_mut();
            for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
                let start = p0.max(p0A);
                let end = p1.min(p1A);
                if start >= end {
                    continue;
                }
                let slc_batch = rt::slice!(start - p0, end - p0);
                let slc_full = rt::slice!(start, end);

                let tmp = -rt::vecdot(j3c_ip2_batch.i((.., .., slc_batch)), rrcd_eri_aux.i((None, None, slc_full)), 2);
                *&mut j1ao_aux1_1.i_mut((.., .., .., A)) += tmp;
            }
        }
        tic!(timing, t0, &format!("evaluate_j3c_ip2, intor contract j-part aux({p0}:{p1})"));

        for iset in 0..nset_k {
            let t0 = std::time::Instant::now();

            let k_in = &k_ins[iset];
            let fold_eri_bra = k_in["fold_eri_bra"].view();
            let mocc_2 = k_in["mocc_2"].view();
            let occ_invsqrt = k_in["occ_invsqrt"].view();
            let nocc = occ_invsqrt.shape()[0];

            // --- j3c_ip2_occ --- //

            let mut j3c_ip2_occ = k_intmds[iset].get_mut("j3c_ip2_occ").unwrap().view_mut();
            let tmp = mocc_2.t() % &j3c_ip2_batch % &mocc_2;
            j3c_ip2_occ.i_mut((.., .., p0..p1)).assign(tmp);

            // --- k1bra_aux1_2 --- //

            let mut k1bra_aux1_2 = k_intmds[iset].get_mut("k1bra_aux1_2").unwrap().view_mut();
            for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
                let start = p0.max(p0A);
                let end = p1.min(p1A);
                if start >= end {
                    continue;
                }
                let slc_batch = rt::slice!(start - p0, end - p0);
                let slc_full = rt::slice!(start, end);

                // NOTE: the following reshape copys the data
                let m1 = fold_eri_bra.i((.., .., slc_full)).swapaxes(0, 1).into_shape((nocc, -1));
                for t in 0..3 {
                    let m2 = j3c_ip2_batch.i((.., .., slc_batch, t)).change_shape((nao, -1));
                    *&mut k1bra_aux1_2.i_mut((.., .., t, A)) -= &m2 % m1.t();
                }
            }

            tic!(timing, t0, &format!("evaluate_j3c_ip2, intor contract k-part {iset} aux({p0}:{p1})"));
        }
    }

    // --- evaluation j-part --- //

    if do_j {
        let t0 = std::time::Instant::now();

        let j_out = j_out.as_mut().unwrap();
        let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();
        let j3c_ip2_aux = j_intmd.as_ref().unwrap()["j3c_ip2_aux"].view();

        // --- dbas evaluation --- //

        let tmp1 = rt::vecdot(&j2c_ip1, &rrcd_eri_aux, 0);
        let dbas_J02_4 = j3c_ip2_aux.i((.., None, None, ..)) * tmp1.i((None, .., .., None)) * &j2c_inv;
        let dbas_J02_5 = j3c_ip2_aux.i((.., None, None, ..)) * j3c_ip2_aux.i((None, .., .., None)) * &j2c_inv;
        let tmp1 = &rrcd_j2c_ip1 * &rrcd_eri_aux;
        let dbas_J02_7 = j3c_ip2_aux.i((.., None, None, ..)) * tmp1.swapaxes(0, 1);

        // --- reduce to hessian contribution --- //

        let mut de_J02_4: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_5: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_7: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_J02_4.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_4.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_J02_5.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_5.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_J02_7.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_7.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_J02_4 = 1.0;
        let scale_J02_5 = 0.5;
        let scale_J02_7 = -1.0;
        j_out.insert("de_J02_4", scale_J02_4 * (&de_J02_4 + &de_J02_4.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_5", scale_J02_5 * (&de_J02_5 + &de_J02_5.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_7", scale_J02_7 * (&de_J02_7 + &de_J02_7.transpose([1, 0, 3, 2])));

        // --- j1ao aux1_2 --- //

        let mut tmp1: Tsr = rt::zeros(([naux, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            tmp1.i_mut((slcA, .., A)).assign(j3c_ip2_aux.i(slcA));
        }
        solve_aux(tmp1.view_mut(), Left, true);
        let j1ao_aux1_2_tp = &cderi % tmp1.reshape((naux, natm * 3));
        let j1ao_aux1_2 = -j1ao_aux1_2_tp.reshape((-1, 3, natm)).unpack_tri(Upper, FlagSymm::Sy);
        j_out.insert("j1ao_aux1_2", j1ao_aux1_2);

        tic!(timing, t0, "evaluate_j3c_ip2, evaluation j-part");
    }

    // --- evaluation k-part --- //

    for iset in 0..nset_k {
        let t0 = std::time::Instant::now();

        let k_in = &k_ins[iset];
        let k_out = &mut k_outs[iset];
        let rrcd_eri_occ = k_in["rrcd_eri_occ"].view();
        let rrcd_eri_bra = k_in["rrcd_eri_bra"].view();
        let occ_invsqrt = k_in["occ_invsqrt"].view();
        let j3c_ip2_occ = k_intmds[iset]["j3c_ip2_occ"].view();
        let nocc = occ_invsqrt.shape()[0];

        // --- dbas evaluation --- //

        let tmp1 = rrcd_eri_occ.reshape([nocc * nocc, naux]) % &j2c_ip1;
        let tmp1 = tmp1.into_shape([nocc, nocc, naux, 3]);
        let j3c_ip2_occ_2d = j3c_ip2_occ.reshape((nocc * nocc, naux, 3));
        let mut dbas_K02_4: Tsr = rt::zeros(([naux, naux, 3, 3], device));
        for t in 0..3 {
            let tmp1_t = tmp1.i((.., .., .., t));
            let tmp1_t = tmp1_t.reshape((nocc * nocc, naux));
            let tmp = tmp1_t.t() % &j3c_ip2_occ_2d * &j2c_inv;
            dbas_K02_4.i_mut((.., .., .., t)).assign(tmp);
        }

        let mut dbas_K02_5: Tsr = rt::zeros(([naux, naux, 3, 3], device));
        for t in 0..3 {
            let tmp1_t = j3c_ip2_occ.i((.., .., .., t));
            let tmp1_t = tmp1_t.reshape((nocc * nocc, naux));
            let tmp = tmp1_t.t() % &j3c_ip2_occ_2d * &j2c_inv;
            dbas_K02_5.i_mut((.., .., .., t)).assign(tmp);
        }

        let rrcd_eri_occ_2d = rrcd_eri_occ.reshape((nocc * nocc, naux));
        let tmp1 = rrcd_eri_occ_2d.t() % j3c_ip2_occ_2d;
        let dbas_K02_7 = tmp1.i((.., .., .., None)) * rrcd_j2c_ip1.i((.., .., None, ..));

        // --- reduce to hessian contribution --- //

        let mut de_K02_4: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_5: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_7: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_K02_4.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_4.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_5.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_5.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_7.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_7.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_K02_4 = 1.0;
        let scale_K02_5 = 0.5;
        let scale_K02_7 = -1.0;
        k_out.insert("de_K02_4", scale_K02_4 * (&de_K02_4 + &de_K02_4.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_5", scale_K02_5 * (&de_K02_5 + &de_K02_5.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_7", scale_K02_7 * (&de_K02_7 + &de_K02_7.transpose([1, 0, 3, 2])));

        // --- k1bra aux1_1 --- //

        let mut k1bra_aux1_1 = rt::zeros(([nao, nocc, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            for t in 0..3 {
                let m1 = rrcd_eri_bra.i((.., .., slcA)).change_shape((nao, -1));
                let m2 = j3c_ip2_occ.i((.., .., slcA, t)).change_shape((nocc, -1));
                k1bra_aux1_1.i_mut((.., .., t, A)).assign(-(&m1 % m2.t()));
            }
        }
        k1bra_aux1_1 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux1_1", k1bra_aux1_1);

        tic!(timing, t0, &format!("evaluate_j3c_ip2, evaluation k-part {iset}"));
    }

    // --- move some intermediates to output --- //

    if do_j {
        let j_out = j_out.as_mut().unwrap();
        let j1ao_aux1_1 = j_intmd.unwrap().remove("j1ao_aux1_1").unwrap();
        j_out.insert("j1ao_aux1_1", j1ao_aux1_1);
    }
    for (iset, k_intmd) in k_intmds.iter_mut().enumerate() {
        let k_out = &mut k_outs[iset];
        let occ_invsqrt = k_ins[iset]["occ_invsqrt"].view();
        let mut k1bra_aux1_2 = k_intmd.remove("k1bra_aux1_2").unwrap();
        *&mut k1bra_aux1_2 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux1_2", k1bra_aux1_2);
    }

    timing
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_j3c_ip1(
    dims: &HashMap<&'static str, usize>,
    shared: &HashMap<&'static str, Tsr>,
    cderi: TsrView,
    mol: &CInt,
    aux: &CInt,
    aoslices: &[[usize; 4]],
    auxslices: &[[usize; 4]],
    aux_ranges: &[[usize; 4]],
    device: &DeviceBLAS,
    solve_aux: &FnSolveAux,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &mut [HashMap<&'static str, Tsr>],
    mut j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
    mut j_intmd: Option<&mut HashMap<&'static str, Tsr>>,
    k_intmds: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let nao = dims["nao"];
    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();
    let j2c_ip1 = shared["j2c_ip1"].view();

    // --- integral contract --- //

    // this part will also handle intermediates, as before in evaluate_j3c_ip2.

    if let Some(j_intmd) = j_intmd.as_mut() {
        j_intmd.insert("j3c_ip1_aux", rt::zeros(([nao, naux, 3], device)));
        j_intmd.insert("j3c_ip1_j1ao_tmp", rt::zeros(([nao, nao, 3], device)));
    }
    for (k_in, k_intmd) in k_ins.iter_mut().zip(k_intmds.iter_mut()) {
        let nocc = k_in["occ_invsqrt"].shape()[0];
        k_intmd.insert("j3c_ip1_bra", rt::zeros(([nao, nocc, naux, 3], device)));
        k_intmd.insert("j3c_ip1_k1ao_tmp", rt::zeros(([nao, nao, 3], device)));
        k_intmd.insert("k1bra_aux0_4", rt::zeros(([nao, nocc, 3, natm], device)));
    }

    for &[sh0, sh1, p0, p1] in aux_ranges.iter() {
        let t0 = std::time::Instant::now();
        let j3c_ip1_batch = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ip1", "s1", device)([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_j3c_ip1, j3c_ip1 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].i(p0..p1);

            // --- j3c_ip1_aux --- //

            let mut j3c_ip1_aux = j_intmd.as_mut().unwrap().get_mut("j3c_ip1_aux").unwrap().view_mut();
            let tmp = rt::vecdot(&j3c_ip1_batch, &dm0, 1);
            j3c_ip1_aux.i_mut((.., p0..p1)).assign(tmp);

            // --- j3c_ip1_j1ao_tmp --- //

            let mut j3c_ip1_j1ao_tmp = j_intmd.as_mut().unwrap().get_mut("j3c_ip1_j1ao_tmp").unwrap().view_mut();
            j3c_ip1_j1ao_tmp += rt::vecdot(&j3c_ip1_batch, rrcd_eri_aux.i((None, None, ..)), 2);
        }
        tic!(timing, t0, &format!("evaluate_j3c_ip1, intor contract j-part aux({p0}:{p1})"));

        for iset in 0..nset_k {
            let t0 = std::time::Instant::now();

            let k_in = &k_ins[iset];
            let k_intmd = &mut k_intmds[iset];
            let rrcd_eri_bra = k_in["rrcd_eri_bra"].view();
            let fold_eri_bra = k_in["fold_eri_bra"].view();
            let mocc_2 = k_in["mocc_2"].view();
            let nocc = k_in["occ_invsqrt"].shape()[0];

            // some tensors are consequently used, need get_disjoint_mut to avoid borrow checker error
            let [j3c_ip1_bra, j3c_ip1_k1ao_tmp] = k_intmd.get_disjoint_mut(["j3c_ip1_bra", "j3c_ip1_k1ao_tmp"]);
            let j3c_ip1_bra = j3c_ip1_bra.unwrap();
            let j3c_ip1_k1ao_tmp = j3c_ip1_k1ao_tmp.unwrap();

            // --- j3c_ip1_bra --- //

            let tmp = &j3c_ip1_batch % &mocc_2;
            j3c_ip1_bra.i_mut((.., .., p0..p1)).assign(tmp);

            // --- j3c_ip1_k1ao_tmp --- //

            let m1 = j3c_ip1_bra.i((.., .., p0..p1, ..)).change_shape((nao, -1, 3));
            let m2 = rrcd_eri_bra.i((.., .., p0..p1)).change_shape((nao, -1));
            *j3c_ip1_k1ao_tmp += m1 % m2.t();

            // --- k1bra_aux0_4 --- //

            let mut k1bra_aux0_4 = k_intmd.get_mut("k1bra_aux0_4").unwrap().view_mut();
            for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
                let slcA = rt::slice!(p0A, p1A);
                // NOTE: reshape with data copy
                let tmp1 = fold_eri_bra.i((slcA, .., p0..p1)).swapaxes(0, 1).into_shape((nocc, -1));
                for t in 0..3 {
                    // NOTE: reshape with data copy
                    let tmp2 = j3c_ip1_batch.i((slcA, .., .., t)).into_swapaxes(0, 1).into_shape((nao, -1));
                    *&mut k1bra_aux0_4.i_mut((.., .., t, A)) -= &tmp2 % tmp1.t();
                }
            }

            tic!(timing, t0, &format!("evaluate_j3c_ip1, intor contract k-part {iset} aux({p0}:{p1})"));
        }
    }

    // --- evaluation j-part --- //

    if do_j {
        let t0 = std::time::Instant::now();

        let j_out = j_out.as_mut().unwrap();
        let j3c_ip1_j1ao_tmp = j_intmd.as_mut().unwrap().remove("j3c_ip1_j1ao_tmp").unwrap();
        let j3c_ip1_aux = j_intmd.as_mut().unwrap().remove("j3c_ip1_aux").unwrap();
        let j3c_ip2_aux = j_intmd.as_mut().unwrap().remove("j3c_ip2_aux").unwrap();
        let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();

        let mut rcd_j3c_ip1_aux = j3c_ip1_aux;
        for t in 0..3 {
            solve_aux(rcd_j3c_ip1_aux.i_mut((.., .., t)), Right, false);
        }

        // --- j1ao aux0 --- //

        let mut j1ao_aux0 = rt::zeros(([nao, nao, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            *&mut j1ao_aux0.i_mut((.., slcA, .., A)) -= j3c_ip1_j1ao_tmp.i(slcA).swapaxes(0, 1);
            *&mut j1ao_aux0.i_mut((slcA, .., .., A)) -= j3c_ip1_j1ao_tmp.i(slcA);
            let tmp1 = rcd_j3c_ip1_aux.i(slcA).sum_axes(0);
            let tmp2 = &cderi % &tmp1;
            *&mut j1ao_aux0.i_mut((Ellipsis, A)) -= 2 * tmp2.unpack_tri(Upper, FlagSymm::Sy);
        }

        j_out.insert("j1ao_aux0", j1ao_aux0);

        // --- dbas evaluation --- //

        let mut dbas_J20_1: Tsr = rt::zeros(([nao, nao, 3, 3], device));
        for t in 0..3 {
            for s in 0..3 {
                let tmp = rcd_j3c_ip1_aux.i((.., .., t)) % rcd_j3c_ip1_aux.i((.., .., s)).t();
                dbas_J20_1.i_mut((.., .., s, t)).assign(tmp);
            }
        }

        let mut rrcd_j3c_ip1_aux = rcd_j3c_ip1_aux;
        for t in 0..3 {
            solve_aux(rrcd_j3c_ip1_aux.i_mut((.., .., t)), Right, true);
        }

        let mut dbas_J11_2: Tsr = rt::zeros(([nao, naux, 3, 3], device));
        for t in 0..3 {
            for s in 0..3 {
                let tmp = rrcd_j3c_ip1_aux.i((.., .., t)) % j2c_ip1.i((.., .., s)).t() * rrcd_eri_aux.i((None, ..));
                dbas_J11_2.i_mut((.., .., s, t)).assign(tmp);
            }
        }

        let tmp1 = rt::vecdot(&j2c_ip1, &rrcd_eri_aux, 0);
        let dbas_J11_3: Tsr = rrcd_j3c_ip1_aux.i((.., .., None, ..)) * tmp1.i((None, .., .., None));
        let dbas_J11_4: Tsr = rrcd_j3c_ip1_aux.i((.., .., None, ..)) * j3c_ip2_aux.i((None, .., .., None));

        // --- reduce to hessian contribution --- //

        let mut de_J20_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J11_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J11_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J11_4: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_J20_1.i((slcA, slcB)).sum_axes([0, 1]);
                de_J20_1.i_mut((.., .., B, A)).assign(tmp);
            }

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_J11_2.i((slcA, slcB)).sum_axes([0, 1]);
                de_J11_2.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J11_3.i((slcA, slcB)).sum_axes([0, 1]);
                de_J11_3.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J11_4.i((slcA, slcB)).sum_axes([0, 1]);
                de_J11_4.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_J20_1 = 2.0;
        let scale_J11_2 = -2.0;
        let scale_J11_3 = 2.0;
        let scale_J11_4 = 2.0;
        j_out.insert("de_J20_1", scale_J20_1 * (&de_J20_1 + &de_J20_1.transpose([1, 0, 3, 2])));
        j_out.insert("de_J11_2", scale_J11_2 * (&de_J11_2 + &de_J11_2.transpose([1, 0, 3, 2])));
        j_out.insert("de_J11_3", scale_J11_3 * (&de_J11_3 + &de_J11_3.transpose([1, 0, 3, 2])));
        j_out.insert("de_J11_4", scale_J11_4 * (&de_J11_4 + &de_J11_4.transpose([1, 0, 3, 2])));

        tic!(timing, t0, "evaluate_j3c_ip1, evaluation j-part");
    }

    // --- evaluation k-part --- //

    for iset in 0..nset_k {
        let t0 = std::time::Instant::now();

        let k_in = &k_ins[iset];
        let k_out = &mut k_outs[iset];
        let rrcd_eri_bra = k_in["rrcd_eri_bra"].view();
        let occ_invsqrt = k_in["occ_invsqrt"].view();
        let fold_eri_bra = k_in["fold_eri_bra"].view();
        let mocc = k_in["mocc"].view();
        let mocc_2 = k_in["mocc_2"].view();
        let nocc = occ_invsqrt.shape()[0];

        let j3c_ip2_occ = k_intmds[iset].remove("j3c_ip2_occ").unwrap();
        let j3c_ip1_bra = k_intmds[iset].remove("j3c_ip1_bra").unwrap();
        let j3c_ip1_k1ao_tmp = k_intmds[iset].remove("j3c_ip1_k1ao_tmp").unwrap();

        // --- k1bra --- //

        let t1 = std::time::Instant::now();

        let mut k1ao_aux_1: Tsr = rt::zeros(([nao, nao, 3, natm], device));
        let mut k1ao_aux_2: Tsr = rt::zeros(([nao, nao, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            *&mut k1ao_aux_1.i_mut((.., slcA, .., A)) -= j3c_ip1_k1ao_tmp.i(slcA).swapaxes(0, 1);
            *&mut k1ao_aux_2.i_mut((slcA, .., .., A)) -= j3c_ip1_k1ao_tmp.i(slcA);
        }
        let k1bra_aux0_1 = k1ao_aux_1 % &mocc;
        let k1bra_aux0_2 = k1ao_aux_2 % &mocc;
        k_out.insert("k1bra_aux0_1", k1bra_aux0_1);
        k_out.insert("k1bra_aux0_2", k1bra_aux0_2);

        let mut k1bra_aux0_3: Tsr = rt::zeros(([nao, nocc, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            for t in 0..3 {
                let tmp1 = j3c_ip1_bra.i((slcA, .., .., t)).swapaxes(0, 1) % mocc.i(slcA);
                let tmp2 = rrcd_eri_bra.reshape((nao, -1)) % tmp1.reshape((nocc, -1)).t();
                k1bra_aux0_3.i_mut((.., .., t, A)).assign(-tmp2);
            }
        }
        k_out.insert("k1bra_aux0_3", k1bra_aux0_3);

        tic!(timing, t1, &format!("evaluate_j3c_ip1, k1bra-1/2/3 {iset}"));

        // --- dbas evaluation rcd_j3c_ip1_bra --- //

        let mut rcd_j3c_ip1_bra = j3c_ip1_bra;
        for t in 0..3 {
            solve_aux(rcd_j3c_ip1_bra.i_mut((.., .., .., t)), Right, false);
        }

        let dm = &mocc_2 % mocc_2.t(); // spin density, not total density

        let t1 = std::time::Instant::now();
        let mut dbas_K20_1a: Tsr = rt::zeros(([nao, nao, 3, 3], device));
        let tmp1 = rcd_j3c_ip1_bra.reshape((nao, nocc * naux, 3));
        for t in 0..3 {
            for s in 0..=t {
                let tmp2 = tmp1.i((.., .., t)) % tmp1.i((.., .., s)).t() * &dm;
                dbas_K20_1a.i_mut((.., .., s, t)).assign(&tmp2);
                // apply symmetric trick
                if t != s {
                    dbas_K20_1a.i_mut((.., .., t, s)).assign(tmp2.t());
                }
            }
        }
        tic!(timing, t1, &format!("evaluate_j3c_ip1, dbas_K20_1a {iset}"));

        // de_K20_1b is special. This term is better to be pre-contracted to hessian.
        let t1 = std::time::Instant::now();
        let mut de_K20_1b: Tsr = rt::zeros(([3, 3, natm, natm], device));
        for &[_, _, p0, p1] in aux_ranges {
            let slcP = rt::slice!(p0, p1);
            let nbatch = p1 - p0;
            let mut fold_occ: Tsr = rt::zeros(([nocc, nocc, nbatch, 3, natm], device));
            for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
                let slcA = rt::slice!(p0A, p1A);
                let tmp1 = rcd_j3c_ip1_bra.i((slcA, .., slcP, ..));
                for t in 0..3 {
                    let tmp = mocc_2.i(slcA).t() % tmp1.i((.., .., .., t));
                    fold_occ.i_mut((.., .., .., t, A)).assign(tmp);
                }
            }
            // non-trivial transpose
            let fold_occ_swap = fold_occ.swapaxes(0, 1).into_contig(ColMajor);
            // handle shape conversions and hessian output
            let fold_occ = fold_occ.reshape([nocc * nocc * nbatch, 3 * natm]);
            let fold_occ_swap = fold_occ_swap.reshape([nocc * nocc * nbatch, 3 * natm]);
            let de_swap = fold_occ.t() % fold_occ_swap; // [s, B, t, A]
            let de_increment = de_swap.reshape([3, natm, 3, natm]).into_swapaxes(1, 2); // [s, t, B, A]
            de_K20_1b += de_increment;
        }
        let scale_K20_1b = 1.0;
        k_out.insert("de_K20_1b", scale_K20_1b * (&de_K20_1b + &de_K20_1b.transpose([1, 0, 3, 2])));
        tic!(timing, t1, &format!("evaluate_j3c_ip1, de_K20_1b {iset}"));

        // --- dbas evaluation rrcd_j3c_ip1_bra --- //

        let mut rrcd_j3c_ip1_bra = rcd_j3c_ip1_bra;
        for t in 0..3 {
            solve_aux(rrcd_j3c_ip1_bra.i_mut((.., .., .., t)), Right, true);
        }

        let t1 = std::time::Instant::now();
        let mut dbas_K11_2: Tsr = rt::zeros(([nao, naux, 3, 3], device));
        for t in 0..3 {
            for s in 0..3 {
                let tmp1 = rrcd_j3c_ip1_bra.i((.., .., .., t)).change_shape([nao * nocc, naux]);
                let tmp2 = (tmp1 % j2c_ip1.i((.., .., s))).into_shape([nao, nocc, naux]);
                let tmp3 = rt::vecdot(&fold_eri_bra, tmp2, 1);
                dbas_K11_2.i_mut((.., .., s, t)).assign(&tmp3);
            }
        }
        tic!(timing, t1, &format!("evaluate_j3c_ip1, dbas_K11_2 {iset}"));

        let t1 = std::time::Instant::now();
        let mut dbas_K11_3: Tsr = rt::zeros(([nao, naux, 3, 3], device));
        for s in 0..3 {
            let tmp1 = fold_eri_bra.reshape([nao * nocc, naux]) % j2c_ip1.i((.., .., s));
            let tmp1 = tmp1.into_shape([nao, nocc, naux]);
            for t in 0..3 {
                let tmp = rt::vecdot(rrcd_j3c_ip1_bra.i((.., .., .., t)), &tmp1, 1);
                dbas_K11_3.i_mut((.., .., s, t)).assign(tmp);
            }
        }
        tic!(timing, t1, &format!("evaluate_j3c_ip1, dbas_K11_3 {iset}"));

        let t1 = std::time::Instant::now();
        let mut dbas_K11_4: Tsr = rt::zeros(([nao, naux, 3, 3], device));
        for s in 0..3 {
            let tmp1 = &mocc_2 % j3c_ip2_occ.i((.., .., .., s));
            for t in 0..3 {
                let tmp = rt::vecdot(rrcd_j3c_ip1_bra.i((.., .., .., t)), &tmp1, 1);
                dbas_K11_4.i_mut((.., .., s, t)).assign(tmp);
            }
        }
        tic!(timing, t1, &format!("evaluate_j3c_ip1, dbas_K11_4 {iset}"));

        // --- reduce to hessian contribution --- //

        let mut de_K20_1a: Tsr = rt::zeros(([3, 3, natm, natm], device));
        // let mut de_K20_1b: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K11_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K11_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K11_4: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_K20_1a.i((slcA, slcB)).sum_axes([0, 1]);
                de_K20_1a.i_mut((.., .., B, A)).assign(tmp);

                // let tmp = dbas_K20_1b.i((slcA, slcB)).sum_axes([0, 1]);
                // de_K20_1b.i_mut((.., .., B, A)).assign(tmp);
            }

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_K11_2.i((slcA, slcB)).sum_axes([0, 1]);
                de_K11_2.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K11_3.i((slcA, slcB)).sum_axes([0, 1]);
                de_K11_3.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K11_4.i((slcA, slcB)).sum_axes([0, 1]);
                de_K11_4.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_K20_1a = 1.0;
        let scale_K11_2 = 2.0;
        let scale_K11_3 = 2.0;
        let scale_K11_4 = 2.0;
        k_out.insert("de_K20_1a", scale_K20_1a * (&de_K20_1a + &de_K20_1a.transpose([1, 0, 3, 2])));
        k_out.insert("de_K11_2", scale_K11_2 * (&de_K11_2 + &de_K11_2.transpose([1, 0, 3, 2])));
        k_out.insert("de_K11_3", scale_K11_3 * (&de_K11_3 + &de_K11_3.transpose([1, 0, 3, 2])));
        k_out.insert("de_K11_4", scale_K11_4 * (&de_K11_4 + &de_K11_4.transpose([1, 0, 3, 2])));

        tic!(timing, t0, &format!("evaluate_j3c_ip1, evaluation k-part {iset}"));
    }

    // --- move some intermediates to output --- //

    for (iset, k_intmd) in k_intmds.iter_mut().enumerate() {
        let k_out = &mut k_outs[iset];
        let occ_invsqrt = k_ins[iset]["occ_invsqrt"].view();
        let mut k1bra_aux0_4 = k_intmd.remove("k1bra_aux0_4").unwrap();
        *&mut k1bra_aux0_4 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux0_4", k1bra_aux0_4);
    }

    timing
}

/* #endregion */

/* #region impl */

/// Generate cderi and decomposition.
pub fn generate_cderi_with_decomp(
    mol: &CInt,
    aux: &CInt,
    j2c_decomp_option: J2CDecompOption,
    device: &DeviceBLAS,
) -> (Tsr, J2CDecompose) {
    let j3c = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s2ij", None, device);
    let j2c_decomp = get_j2c_decomp(aux, device, j2c_decomp_option);
    let cderi = solve_by_j2c(j3c, &j2c_decomp, Right, false);
    (cderi, j2c_decomp)
}

pub struct RHessRIJK<'a> {
    pub mol: CInt,
    pub aux: CInt,
    pub factor_j: f64,
    pub factor_k: f64,
    pub cderi: TsrCow<'a>,
    pub j2c_decomp: J2CDecompose,
    pub intmd: HashMap<String, Tsr>, // intermediates
    pub result: HashMap<&'static str, Tsr>,
    pub timing: Vec<(String, f64)>,
    pub is_skeleton_ready: bool,
}

impl<'a> RHessRIJK<'a> {
    pub fn new_without_cderi(mol: &CInt, aux: &CInt, factor_j: f64, factor_k: f64, j2c_decomp_option: J2CDecompOption) -> Self {
        // note: following options should be valid
        // let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14), uplo: Upper };
        // let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14), uplo: Lower };
        // let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Eig, threshold: Some(1e-14), uplo: Upper };
        let device = DeviceBLAS::default();
        let (cderi, j2c_decomp) = generate_cderi_with_decomp(mol, aux, j2c_decomp_option, &device);
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            factor_j,
            factor_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
            timing: Vec::new(),
            is_skeleton_ready: false,
        }
    }

    pub fn new_with_cderi(
        mol: &CInt,
        aux: &CInt,
        factor_j: f64,
        factor_k: f64,
        cderi: TsrCow<'a>,
        j2c_decomp: J2CDecompose,
    ) -> Self {
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            factor_j,
            factor_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
            timing: Vec::new(),
            is_skeleton_ready: false,
        }
    }

    pub fn ensure_skeleton(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) {
        if self.is_skeleton_ready {
            return;
        }
        let (j_out, k_outs, timing) = get_rijk_skeleton_decomposed_separated(
            &self.mol,
            &self.aux,
            &[mo_coeff],
            &[mo_occ],
            self.cderi.view(),
            &self.j2c_decomp,
            self.factor_j != 0.0,
            self.factor_k != 0.0,
            72, // TODO: batch size `72` should be tunable by max-memory.
            atm_list,
            None,
        );
        self.timing.extend(timing);

        if let Some(j_out) = j_out {
            for (key, value) in j_out.into_iter() {
                self.intmd.insert(key.to_string(), value);
            }
        };

        for (iset, k_out) in k_outs.into_iter().enumerate() {
            // note the keys can clash for output of k.
            // for storage of intermediates, we append `<spin_{iset}>` to the key name.
            for (key, value) in k_out.into_iter() {
                self.intmd.insert(format!("{key}<spin_{iset}>"), value);
            }
        }

        self.is_skeleton_ready = true;
    }
}

impl<'a> AnalDrvBaseAPI for RHessRIJK<'a> {}

impl<'a> RHessElecInteractAPI for RHessRIJK<'a> {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        self.ensure_skeleton(mo_coeff, mo_occ, atm_list);
        let intmd = &self.intmd;

        let device = self.cderi.device();
        let natm = atm_list.map_or_else(|| self.mol.natm(), |list| list.len());
        let hess_init = || -> Tsr { rt::zeros(([3, 3, natm, natm], device)) };

        let mut de = hess_init();
        if self.factor_j != 0.0 {
            let de_J20 = KEYS_J20.iter().map(|&key| &intmd[key]).fold(hess_init(), |acc, x| acc + x);
            let de_J11 = KEYS_J11.iter().map(|&key| &intmd[key]).fold(hess_init(), |acc, x| acc + x);
            let de_J02 = KEYS_J02.iter().map(|&key| &intmd[key]).fold(hess_init(), |acc, x| acc + x);
            let de_J = &de_J20 + &de_J11 + &de_J02;
            de += self.factor_j * &de_J;
            self.result.insert("de_J20", de_J20);
            self.result.insert("de_J11", de_J11);
            self.result.insert("de_J02", de_J02);
            self.result.insert("de_J", de_J);
        }
        if self.factor_k != 0.0 {
            // rhf only have one spin
            let de_K20 =
                KEYS_K20.iter().map(|&key| &intmd[&format!("{key}<spin_0>")]).fold(hess_init(), |acc, x| acc + x);
            let de_K11 =
                KEYS_K11.iter().map(|&key| &intmd[&format!("{key}<spin_0>")]).fold(hess_init(), |acc, x| acc + x);
            let de_K02 =
                KEYS_K02.iter().map(|&key| &intmd[&format!("{key}<spin_0>")]).fold(hess_init(), |acc, x| acc + x);
            let de_K = &de_K20 + &de_K11 + &de_K02;
            de -= 0.5 * self.factor_k * &de_K;
            self.result.insert("de_K20", de_K20);
            self.result.insert("de_K11", de_K11);
            self.result.insert("de_K02", de_K02);
            self.result.insert("de_K", de_K);
        }
        self.result.insert("de_skeleton", de.clone());
        de
    }

    fn get_deriv1_ao(&mut self, _mo_coeff: TsrView, _mo_occ: TsrView, _atm_list: Option<&[usize]>) -> Tsr {
        unimplemented!("This function is not implemented for optimized RI-JK hessian. Use `get_deriv1_bra` instead.")
    }

    fn get_deriv1_bra(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        self.ensure_skeleton(mo_coeff.view(), mo_occ.view(), atm_list);
        let intmd = &self.intmd;

        let device = self.cderi.device();
        let natm = atm_list.map_or_else(|| self.mol.natm(), |list| list.len());
        let nao = mo_coeff.shape()[0];
        let occidx = mo_occ.greater(0.0).into_vec();
        let nocc = occidx.iter().filter(|&&x| x).count();
        let mocc = mo_coeff.bool_select(-1, &occidx).into_contig(ColMajor);

        let deriv1_ao_init = || -> Tsr { rt::zeros(([nao, nao, 3, natm], device)) };
        let deriv1_bra_init = || -> Tsr { rt::zeros(([nao, nocc, 3, natm], device)) };

        let mut deriv1_bra = deriv1_bra_init();
        if self.factor_j != 0.0 {
            let j1ao = KEYS_J1AO.iter().map(|&key| &intmd[key]).fold(deriv1_ao_init(), |acc, x| acc + x);
            deriv1_bra += self.factor_j * (&j1ao % &mocc);
            self.result.insert("j1ao", j1ao);
        }
        if self.factor_k != 0.0 {
            let k1bra = KEYS_K1BRA
                .iter()
                .map(|&key| &intmd[&format!("{key}<spin_0>")])
                .fold(deriv1_bra_init(), |acc, x| acc + x);
            deriv1_bra -= 0.5 * self.factor_k * &k1bra;
            self.result.insert("k1bra", k1bra);
        }
        self.result.insert("deriv1_bra", deriv1_bra.clone());
        deriv1_bra
    }
}

/* #endregion */
