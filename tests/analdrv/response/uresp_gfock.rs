//! Unrestricted response-kernel rdm form (`get_response_rdm`, the ugfock groundwork) for
//! URespRIJK and URespKSNIMatmul, pinned three ways:
//! - consistency with the hessian-tested bra form (`get_response_bra`): the rdm form on the
//!   symmetrized bra-generated densities, right half-transformed by `mocc`, equals the bra form;
//! - closed-shell limit: with both spins carrying the restricted test density, each spin's
//!   response equals the restricted kernel (`4 J - 2 K` / `4 fxc`);
//! - precision plumbing: with the low-precision resource (`cderi_resp` / `ni_resp`) set to a
//!   duplicate of the high-precision one, every form agrees between `prec = true` and
//!   `prec = false`.

use pyrest::analdrv::response::rresp_interface::{scf_jk_factors, scf_xc_func_list};
use pyrest::analdrv::response::trait_rresp::RRespAPI;
use pyrest::analdrv::response::trait_uresp::URespAPI;
use pyrest::analdrv::response::uresp_interface::scf_xc_func_list_uks;
use pyrest::ctrl_io;
use pyrest::dft::numint_matmul::nimatmul::NIMatmul;
use pyrest::dft::numint_matmul::resp_rks::RRespKSNIMatmul;
use pyrest::dft::numint_matmul::resp_uks::URespKSNIMatmul;
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::resp_r::RRespRIJK;
use pyrest::ri_jk::resp_u::URespRIJK;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI, Tsr};

use rstsr::prelude::*;

static INPUT_OH_UKS: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "b3lyp"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 2.0
    spin_polarization =    true

[geom]
    name = "OH"
    unit = "Angstrom"
    position = """
    O  0.0   0.0   0.0
    H  0.75  0.55  0.10
    """
"##;

static INPUT_H2O_RKS: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "b3lyp"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 1.0
    spin_polarization =    false

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
    O  0.0   0.0   0.0
    H  0.75  0.55  0.10
    H -0.70  0.60 -0.05
    """
"##;

fn build_scf(input: &str) -> scf_io::SCF {
    let keys = toml::from_str::<serde_json::Value>(input).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    scf_data
}

fn det_matrix(m: usize, n: usize, seed: usize, device: &DeviceBLAS) -> Tsr {
    let data: Vec<f64> = (0..m * n).map(|k| (((k + seed) * 7 + 3) % 11) as f64 / 11.0 - 0.5).collect();
    rt::asarray((data, vec![m, n], device))
}

#[test]
fn test_oh_uks_response_rdm_consistency() {
    let scf_data = build_scf(INPUT_OH_UKS);
    let device = rt::DeviceBLAS::default();
    let mo_coeff = [(&scf_data.eigenvectors[0]).to_rstsr(&device), (&scf_data.eigenvectors[1]).to_rstsr(&device)];
    let mo_occ = [(&scf_data.occupation[0]).to_rstsr(&device), (&scf_data.occupation[1]).to_rstsr(&device)];
    let nao = mo_coeff[0].shape()[0];

    // --- URespRIJK: bra form vs rdm form on the bra-generated densities --- //
    let (factor_j, factor_k, _) = scf_jk_factors(&scf_data);
    let (rimatr, _, _) = scf_data.rimatr.as_ref().unwrap();
    let mut rijk = URespRIJK::new_with_cderi(factor_j, factor_k, rimatr.to_rstsr_view(&device).into_cow());
    rijk.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], true);

    let occidx = [mo_occ[0].view().greater(0).into_vec(), mo_occ[1].view().greater(0).into_vec()];
    let mocc = [mo_coeff[0].bool_select(-1, &occidx[0]), mo_coeff[1].bool_select(-1, &occidx[1])];
    let bra = [det_matrix(nao, mocc[0].shape()[1], 1, &device), det_matrix(nao, mocc[1].shape()[1], 2, &device)];

    let resp_bra = rijk.get_response_bra(&[bra[0].view(), bra[1].view()], true);
    let x = [&bra[0] % mocc[0].t(), &bra[1] % mocc[1].t()];
    let x_sym = [(&x[0] + &x[0].t()) * 0.5, (&x[1] + &x[1].t()) * 0.5];
    let resp_rdm = rijk.get_response_rdm(&[x_sym[0].view(), x_sym[1].view()], true);
    for s in 0..2 {
        let expect = resp_rdm[s].view() % mocc[s].view();
        assert!(rt::allclose(&resp_bra[s], &expect, (1e-8, 1e-9)), "RI-JK bra/rdm mismatch, spin {s}");
    }

    // --- URespKSNIMatmul: the same check on the XC kernel --- //
    let grids = scf_data.grids.as_ref().unwrap();
    let mol_cint = get_cint_mol(&scf_data.mol);
    let ni = NIMatmul::new(&mol_cint, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);
    let mut uks = URespKSNIMatmul::new(scf_xc_func_list_uks(&scf_data), ni, false);
    uks.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], true);

    let resp_bra = uks.get_response_bra(&[bra[0].view(), bra[1].view()], true);
    let resp_rdm = uks.get_response_rdm(&[x_sym[0].view(), x_sym[1].view()], true);
    for s in 0..2 {
        let expect = resp_rdm[s].view() % mocc[s].view();
        assert!(rt::allclose(&resp_bra[s], &expect, (1e-7, 1e-8)), "UKS bra/rdm mismatch, spin {s}");
    }
}

#[test]
fn test_closed_shell_limit() {
    let scf_data = build_scf(INPUT_H2O_RKS);
    let device = rt::DeviceBLAS::default();
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
    let nao = mo_coeff.shape()[0];

    // deterministic symmetric test density
    let raw = det_matrix(nao, nao, 5, &device);
    let x = (&raw + &raw.t()) * 0.5;

    // --- RI-JK: U(X, X) per spin == R(X) --- //
    let (factor_j, factor_k, _) = scf_jk_factors(&scf_data);
    let (rimatr, _, _) = scf_data.rimatr.as_ref().unwrap();
    let mut rrijk = RRespRIJK::new_with_cderi(factor_j, factor_k, rimatr.to_rstsr_view(&device).into_cow());
    let mut urijk = URespRIJK::new_with_cderi(factor_j, factor_k, rimatr.to_rstsr_view(&device).into_cow());
    let resp_r = rrijk.get_response_rdm(x.view(), true);
    let resp_u = urijk.get_response_rdm(&[x.view(), x.view()], true);
    for s in 0..2 {
        assert!(rt::allclose(&resp_u[s], &resp_r, (1e-8, 1e-9)), "RI-JK closed-shell mismatch, spin {s}");
    }

    // --- KS XC: the same, with the U object prepared at halved occupation (occ 1 per spin) --- //
    let grids = scf_data.grids.as_ref().unwrap();
    let mol_cint = get_cint_mol(&scf_data.mol);
    let ni_r = NIMatmul::new(&mol_cint, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);
    let ni_u = NIMatmul::new(&mol_cint, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);
    let mut rks = RRespKSNIMatmul::new(scf_xc_func_list(&scf_data), ni_r, false);
    let mut uks = URespKSNIMatmul::new(scf_xc_func_list_uks(&scf_data), ni_u, false);
    rks.make_response_preparation(mo_coeff.view(), mo_occ.view(), true);
    let occ_half = &mo_occ * 0.5_f64;
    uks.make_response_preparation(&[mo_coeff.view(), mo_coeff.view()], &[occ_half.view(), occ_half.view()], true);

    let resp_r = rks.get_response_rdm(x.view(), true);
    let resp_u = uks.get_response_rdm(&[x.view(), x.view()], true);
    for s in 0..2 {
        assert!(rt::allclose(&resp_u[s], &resp_r, (1e-7, 1e-8)), "UKS closed-shell mismatch, spin {s}");
    }
}

/// Precision plumbing: with the low-precision resource (`cderi_resp` of `URespRIJK`, `ni_resp`
/// of `URespKSNIMatmul`) set to a duplicate of the high-precision one, every form agrees between
/// `prec = true` and `prec = false` — and the attached object agrees with an unattached one.
#[test]
fn test_precision_selection_plumbing() {
    let scf_data = build_scf(INPUT_OH_UKS);
    let device = rt::DeviceBLAS::default();
    let mo_coeff = [(&scf_data.eigenvectors[0]).to_rstsr(&device), (&scf_data.eigenvectors[1]).to_rstsr(&device)];
    let mo_occ = [(&scf_data.occupation[0]).to_rstsr(&device), (&scf_data.occupation[1]).to_rstsr(&device)];
    let nao = mo_coeff[0].shape()[0];

    let occidx = [mo_occ[0].view().greater(0).into_vec(), mo_occ[1].view().greater(0).into_vec()];
    let mocc = [mo_coeff[0].bool_select(-1, &occidx[0]), mo_coeff[1].bool_select(-1, &occidx[1])];
    let bra = [det_matrix(nao, mocc[0].shape()[1], 1, &device), det_matrix(nao, mocc[1].shape()[1], 2, &device)];
    let x = [&bra[0] % mocc[0].t(), &bra[1] % mocc[1].t()];
    let x_sym = [(&x[0] + &x[0].t()) * 0.5, (&x[1] + &x[1].t()) * 0.5];

    // --- URespRIJK: cderi_resp as a second view of the same rimatr --- //
    let (factor_j, factor_k, _) = scf_jk_factors(&scf_data);
    let (rimatr, _, _) = scf_data.rimatr.as_ref().unwrap();
    let mut rijk = URespRIJK::new_with_cderi(factor_j, factor_k, rimatr.to_rstsr_view(&device).into_cow())
        .set_cderi_resp(rimatr.to_rstsr_view(&device).into_cow());
    let mut rijk_ref = URespRIJK::new_with_cderi(factor_j, factor_k, rimatr.to_rstsr_view(&device).into_cow());

    rijk.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], true);
    let bra_high = rijk.get_response_bra(&[bra[0].view(), bra[1].view()], true);
    let rdm_high = rijk.get_response_rdm(&[x_sym[0].view(), x_sym[1].view()], true);
    let fock_high = rijk.get_fock_rdm(&[x_sym[0].view(), x_sym[1].view()], true);
    rijk.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], false);
    let bra_low = rijk.get_response_bra(&[bra[0].view(), bra[1].view()], false);
    let rdm_low = rijk.get_response_rdm(&[x_sym[0].view(), x_sym[1].view()], false);
    let fock_low = rijk.get_fock_rdm(&[x_sym[0].view(), x_sym[1].view()], false);

    rijk_ref.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], true);
    let bra_ref = rijk_ref.get_response_bra(&[bra[0].view(), bra[1].view()], true);

    for s in 0..2 {
        assert!(rt::allclose(&bra_low[s], &bra_high[s], (1e-10, 1e-12)), "RI-JK bra precision mismatch, spin {s}");
        assert!(rt::allclose(&rdm_low[s], &rdm_high[s], (1e-10, 1e-12)), "RI-JK rdm precision mismatch, spin {s}");
        assert!(rt::allclose(&fock_low[s], &fock_high[s], (1e-10, 1e-12)), "RI-JK fock precision mismatch, spin {s}");
        assert!(rt::allclose(&bra_high[s], &bra_ref[s], (1e-10, 1e-12)), "RI-JK attached-vs-plain mismatch, spin {s}");
    }

    // --- URespKSNIMatmul: ni_resp as a second driver over the same grid --- //
    let grids = scf_data.grids.as_ref().unwrap();
    let mol_cint = get_cint_mol(&scf_data.mol);
    let ni = NIMatmul::new(&mol_cint, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);
    let ni_resp = NIMatmul::new(&mol_cint, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);
    let mut uks = URespKSNIMatmul::new(scf_xc_func_list_uks(&scf_data), ni, false).set_ni_resp(ni_resp);

    uks.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], true);
    let bra_high = uks.get_response_bra(&[bra[0].view(), bra[1].view()], true);
    let rdm_high = uks.get_response_rdm(&[x_sym[0].view(), x_sym[1].view()], true);
    uks.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()], false);
    let bra_low = uks.get_response_bra(&[bra[0].view(), bra[1].view()], false);
    let rdm_low = uks.get_response_rdm(&[x_sym[0].view(), x_sym[1].view()], false);

    for s in 0..2 {
        assert!(rt::allclose(&bra_low[s], &bra_high[s], (1e-10, 1e-12)), "UKS bra precision mismatch, spin {s}");
        assert!(rt::allclose(&rdm_low[s], &rdm_high[s], (1e-10, 1e-12)), "UKS rdm precision mismatch, spin {s}");
    }
}
