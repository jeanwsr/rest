//! Validation harness for the **zero-copy** AO-basis GW route
//! (`gw_tensor_style = "ao"`, `src/ri_gw/tensor_ao.rs`).
//!
//! The route must satisfy two contracts:
//!
//! 1. **values**: every tensor it produces equals the historical MO-basis one
//!    (`ri3mo` rows, `V`, `W_c`, the RPA response matrix, the contour residue,
//!    and finally every quasiparticle energy);
//! 2. **memory**: it must not materialise `ri_ov` nor copy `SCF::rimatr`.
//!    `tests/bench_gw_memory_peaks.rs` measures the second contract; here it is
//!    asserted structurally (`GwTensorRoute::ri_ov` returns `None`, the plan
//!    holds no RI data, `SCF::rimatr` stays owned by the `SCF`).
//!
//! Derivation: `docs/gw_ao_lean_derivation.md`.

use pyrest::ctrl_io::quasiparticle_methods::QuasiParticle;
use pyrest::molecule_io::Molecule;
use pyrest::ri_gw::tensor_ao::{self, GwAoPlan};
use pyrest::ri_gw::GwTensorRoute;
use pyrest::ri_bse;
use pyrest::scf_io::{self, SCF};
use rest_tensors::MatrixFull;

use std::time::Instant;

// ---------------------------------------------------------------- scaffolding

fn input_for(geom: &str, aux: &str) -> String {
    format!(
        r##"
[ctrl]
print_level = 0
num_threads = 4
xc = "pbe0"
basis_path = "cc-pvdz"
basis_type = "spheric"
auxbas_path = "{aux}"
auxbas_type = "spheric"
eri_type = "ri-v"
use_ri_symm = true
charge = 0.0
spin = 1.0
spin_polarization = false
initial_guess = "sad"
mixer = "diis"
max_scf_cycle = 100
scf_acc_rho = 1.0e-8
scf_acc_eev = 1.0e-7
scf_acc_etot = 1.0e-10
[geom]
{geom}
[quasiparticle_methods]
gw_or_bse = "gw"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
gw_extrapolate_occ_threshold = 100.0
gw_extrapolate_vir_threshold = 100.0
bse_cutoff_energy = 1.0e6
"##
    )
}

fn nh3_geom() -> &'static str {
    r#"
name = "NH3"
unit = "Angstrom"
position = """
    N   0.00000000   0.00000000   0.11730000
    H   0.00000000   0.93770000  -0.27370000
    H   0.81210000  -0.46890000  -0.27370000
    H  -0.81210000  -0.46890000  -0.27370000
"""
"#
}

fn h2o_geom() -> &'static str {
    r#"
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11730000
    H   0.00000000   0.75720000  -0.46920000
    H   0.00000000  -0.75720000  -0.46920000
"""
"#
}

fn qp_default() -> QuasiParticle {
    QuasiParticle {
        gw_scheme: "extrapolated".to_string(),
        scgw: "g0w0".to_string(),
        gw_or_bse: "gw".to_string(),
        use_low_rank_contour: false,
        gw_extrapolate_occ_threshold: 100.0,
        gw_extrapolate_vir_threshold: 100.0,
        gw_tensor_style: "mo".to_string(),
        ..Default::default()
    }
}

fn build_with(geom: &str, aux: &str, qp: QuasiParticle) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(&input_for(geom, aux)[..]).unwrap();
    let (mut ctrl, geom) = pyrest::ctrl_io::parse_ctl_from_json(&keys).unwrap();
    ctrl.quasiparticle_methods = Some(qp);
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf
}

fn build(geom: &str, aux: &str) -> SCF {
    build_with(geom, aux, qp_default())
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let scale = max_abs(a).max(max_abs(b)).max(1e-300);
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

fn mf_diff(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> f64 {
    assert_eq!(a.size, b.size, "shape mismatch {:?} vs {:?}", a.size, b.size);
    rel_diff(&a.data, &b.data)
}

/// The plan must not own any RI-sized array, and no `ri_ov` may appear.
#[test]
fn plan_is_zero_copy() {
    let mut scf = build(nh3_geom(), "def2-universal-jkfit");
    let n_packed = scf.mol.num_basis * (scf.mol.num_basis + 1) / 2;
    let n_aux = scf.rimatr.as_ref().unwrap().0.size[1];
    let ao_tensor_bytes = n_packed * n_aux * 8;

    scf.mol.ctrl.quasiparticle_methods.as_mut().unwrap().gw_tensor_style = "ao".to_string();
    let route = GwTensorRoute::select(&mut scf);
    assert!(route.is_ao());

    let plan = route.plan.as_ref().unwrap();
    assert!(scf.rimatr.is_some(), "SCF::rimatr must keep ownership");
    assert!(
        plan.bytes < ao_tensor_bytes / 100 || plan.bytes == 0,
        "plan holds {} bytes vs {} for the AO tensor",
        plan.bytes,
        ao_tensor_bytes
    );
    assert!(route.ri_ov(&scf).is_none(), "the AO route must not build ri_ov");
    println!(
        "plan bookkeeping = {} bytes, AO tensor = {} bytes ({:.1} MB); ri_ov = None",
        plan.bytes,
        ao_tensor_bytes,
        ao_tensor_bytes as f64 / 1048576.0
    );

    let mut scf_mo = build(nh3_geom(), "def2-universal-jkfit");
    let route_mo = GwTensorRoute::select(&mut scf_mo);
    assert!(!route_mo.is_ao());
    assert!(route_mo.ri_ov(&scf_mo).is_some());
}

#[test]
fn ao_rows_and_ri_ov_match_mo() {
    let scf = build(nh3_geom(), "def2-universal-jkfit");
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let plan = GwAoPlan::build(&scf).expect("plan should build");
    assert_eq!(plan.dims.n_o, n_o);
    assert_eq!(plan.dims.n_v, n_v);
    assert_eq!(plan.dims.n_mo, num_state - start_mo);

    let mut worst = 0.0_f64;
    for n in 0..plan.dims.n_mo {
        let m = pyrest::ri_gw::compute_ri3mo_row(&scf, n);
        let a = tensor_ao::ri_row(&scf, &plan, start_mo + n, start_mo, plan.dims.n_mo);
        worst = worst.max(mf_diff(&m, &a));
    }
    println!("all {} ri3mo rows: max rel diff = {:.3e}", plan.dims.n_mo, worst);
    assert!(worst < 1e-12, "ri3mo rows mismatch: {:.3e}", worst);

    let mo = ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    let ao = tensor_ao::ri_ov_materialised(&scf, &plan);
    let e = mf_diff(&mo, &ao);
    println!("ri_ov (test-only materialisation): max rel diff = {:.3e}", e);
    assert!(e < 1e-12, "ri_ov mismatch: {:.3e}", e);
}

/// The heart of the redesign: the response matrix accumulated as a sum of
/// rank-`V` updates must equal `ri_ov diag(f) ri_ov^T`.
#[test]
fn ao_response_matrix_matches_mo() {
    let scf = build(nh3_geom(), "def2-universal-jkfit");
    let (_, _, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let plan = GwAoPlan::build(&scf).unwrap();
    let ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    let qp_w = scf.eigenvalues[0].clone();

    for (label, part, omega, eta) in [
        ("I", 'I', 0.7_f64, 0.0_f64),
        ("C", 'C', 0.25_f64, 0.0_f64),
        ("C-eta", 'C', 0.25_f64, 0.01_f64),
    ] {
        let t = Instant::now();
        let r_mo = pyrest::ri_gw::response_matrix(&qp_w, n_o, n_v, &ri_ov, omega, part, eta);
        let dt_mo = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let r_ao = tensor_ao::response_matrix(&scf, &plan, &qp_w, omega, part, eta);
        let dt_ao = t.elapsed().as_secs_f64();
        let e = mf_diff(&r_mo, &r_ao);
        println!(
            "response_matrix[{}]: max rel diff {:.3e}   (MO {:.1} ms, AO {:.1} ms)",
            label,
            e,
            dt_mo * 1e3,
            dt_ao * 1e3
        );
        assert!(e < 1e-12, "response matrix ({}) mismatch: {:.3e}", label, e);
    }
}

#[test]
fn ao_v_and_w_c_match_mo() {
    let scf = build(nh3_geom(), "def2-universal-jkfit");
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let plan = GwAoPlan::build(&scf).unwrap();
    let n_mo = num_state - start_mo;

    let v_mo = pyrest::ri_gw::v_matrix_from_scf(&scf);
    let v_ao = tensor_ao::v_matrix(&scf, &plan);
    let mut a = vec![0.0; n_mo * n_mo];
    let mut b = vec![0.0; n_mo * n_mo];
    for n in 0..n_mo {
        for m in 0..n_mo {
            a[n + m * n_mo] = v_mo[[n, m]];
            b[n + m * n_mo] = v_ao[[n, m]];
        }
    }
    let e = rel_diff(&a, &b);
    println!("V[n,m]: max rel diff = {:.3e}", e);
    assert!(e < 1e-12, "V mismatch: {:.3e}", e);

    let qp_w = scf.eigenvalues[0].clone();
    let ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    for (label, part, omega) in [("I", 'I', 0.7_f64), ("C", 'C', 0.25_f64)] {
        let response = pyrest::ri_gw::response_matrix(&qp_w, n_o, n_v, &ri_ov, omega, part, 0.0);
        let inv = pyrest::ri_gw::inverse_dielectric_matrix(response, part);
        let w_mo = pyrest::ri_gw::w_c_matrix_from_scf(&inv, num_state, &scf);
        let mut w_ao = vec![MatrixFull::new([num_state, num_state], 0.0)];
        tensor_ao::w_c_matrices(&scf, &plan, std::slice::from_ref(&inv), &mut w_ao);
        let mut a = vec![0.0; n_mo * n_mo];
        let mut b = vec![0.0; n_mo * n_mo];
        for n in 0..n_mo {
            for m in 0..n_mo {
                a[n + m * n_mo] = w_mo[[n, m]];
                b[n + m * n_mo] = w_ao[0][[n, m]];
            }
        }
        let e = rel_diff(&a, &b);
        println!("W_c[{}]: max rel diff = {:.3e}", label, e);
        assert!(e < 1e-12, "W_c ({}) mismatch: {:.3e}", label, e);
    }
}

#[test]
fn ao_contour_residue_matches_mo() {
    let scf = build(h2o_geom(), "def2-universal-jkfit");
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let plan = GwAoPlan::build(&scf).unwrap();
    let mo_ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');

    let qp_g = scf.eigenvalues[0].clone();
    let qp_w = qp_g.clone();
    let res_tol = 1e-6_f64;
    let eta = 0.0_f64;

    for n in [0usize, 1, n_o, n_o + 1, num_state - 1] {
        let mo_row = pyrest::ri_gw::compute_ri3mo_row(&scf, n);
        let ao_row = tensor_ao::ri_row(&scf, &plan, start_mo + n, start_mo, plan.dims.n_mo);
        for omega in [qp_g[n] - 0.1, qp_g[n], qp_g[n] + 0.1] {
            let a = pyrest::ri_gw::contour_rayon(
                omega, n, &qp_g, &qp_w, n_o, n_v, num_state, &mo_ri_ov, &mo_row, res_tol, eta,
            );
            let b = pyrest::ri_gw::contour_rayon_ao(
                &scf, &plan, omega, n, &qp_g, &qp_w, n_o, n_v, num_state, &ao_row, res_tol, eta,
            );
            assert!(
                (a - b).abs() < 1e-10,
                "contour mismatch at n={} omega={}: {} vs {}",
                n,
                omega,
                a,
                b
            );
        }
    }
    println!("contour residues agree for 5 orbitals x 3 frequencies");
}

/// End-to-end: the quasiparticle energies of the two routes must be identical.
#[test]
fn ao_qp_energies_match_mo() {
    let mut scf_mo = build(nh3_geom(), "def2-universal-jkfit");
    let mut scf_ao = scf_mo.clone();
    scf_ao.mol.ctrl.quasiparticle_methods.as_mut().unwrap().gw_tensor_style = "ao".to_string();

    pyrest::ri_gw::initialize_qp_g_w(&mut scf_mo);
    pyrest::ri_gw::initialize_qp_g_w(&mut scf_ao);
    let vxc = pyrest::ri_gw::vxc_ao2mo(&scf_mo);
    let num_freq = 8;

    let qp_mo =
        pyrest::ri_gw::scgw::gw_near_fermi_surface(&mut scf_mo, num_freq, &vxc, 1.0e6, 1.0e6);
    let qp_ao =
        pyrest::ri_gw::scgw::gw_near_fermi_surface(&mut scf_ao, num_freq, &vxc, 1.0e6, 1.0e6);
    assert_eq!(qp_mo.len(), qp_ao.len());
    let d = qp_mo
        .iter()
        .zip(qp_ao.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "QP energies (MO vs AO): max |diff| = {:.3e} Ha [{} orbitals]",
        d,
        qp_mo.len()
    );
    for (n, (a, b)) in qp_mo.iter().zip(qp_ao.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "QP energy {} differs: {} vs {}", n, a, b);
    }
}

#[test]
fn screening_keeps_all_pairs_at_zero_tol() {
    let scf = build(nh3_geom(), "def2-universal-jkfit");
    let base = GwAoPlan::build_with_screening(&scf, 0.0).unwrap();
    assert_eq!(base.kept_pairs, base.dims.n_packed);
    assert!(!base.screening_active());

    let screened = GwAoPlan::build_with_screening(&scf, 2.0e-3).unwrap();
    println!(
        "tol=2e-3 keeps {}/{} pairs ({:.1}%)",
        screened.kept_pairs,
        screened.dims.n_packed,
        100.0 * screened.kept_pairs as f64 / screened.dims.n_packed as f64
    );
    assert!(screened.kept_pairs <= screened.dims.n_packed);
    assert!(screened.screening_active());

    let mo = ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    let e0 = mf_diff(&mo, &tensor_ao::ri_ov_materialised(&scf, &base));
    let e1 = mf_diff(&mo, &tensor_ao::ri_ov_materialised(&scf, &screened));
    println!("ri_ov rel error: unscreened {:.3e}, tol=2e-3 {:.3e}", e0, e1);
    assert!(e1 >= e0 * 0.5);
}

#[test]
fn route_selection_and_cache() {
    let mut scf = build(nh3_geom(), "def2-universal-jkfit");
    let r = GwTensorRoute::select(&mut scf);
    assert!(!r.is_ao(), "default style must be the MO route");

    scf.mol.ctrl.quasiparticle_methods.as_mut().unwrap().gw_tensor_style = "ao".to_string();
    let r = GwTensorRoute::select(&mut scf);
    assert!(r.is_ao());
    assert!(scf.gw_ao_ctx.is_some());

    let t0 = Instant::now();
    let r2 = GwTensorRoute::select(&mut scf);
    let dt = t0.elapsed().as_secs_f64();
    assert!(r2.is_ao());
    println!("cached plan lookup took {:.6} s", dt);
    assert!(dt < 0.5, "cache miss: {:.3}s", dt);

    // the low-rank contour still requires the MO route
    scf.mol.ctrl.quasiparticle_methods.as_mut().unwrap().use_low_rank_contour = true;
    let r3 = GwTensorRoute::select(&mut scf);
    assert!(!r3.is_ao());
}

/// The streaming `u^(i)` fold (packed-triangle sweep + one `dgemm`) must equal
/// the block fold used everywhere else.
#[test]
fn streaming_u_fold_matches_block_fold() {
    let scf = build(nh3_geom(), "def2-universal-jkfit");
    let plan = GwAoPlan::build(&scf).unwrap();
    let n_o = plan.dims.n_o;
    for i in 0..n_o {
        let a = tensor_ao::occ_vir_block(&scf, &plan, i);
        let b = tensor_ao::mo_block_matfull(
            &scf,
            &plan,
            &[plan.dims.start_mo + i],
            plan.dims.lumo,
            plan.dims.n_v,
        );
        let e = mf_diff(&b, &a);
        assert!(e < 1e-12, "u^(i={}) mismatch: {:.3e}", i, e);
    }
    println!("all {} u^(i) blocks agree with the block fold", n_o);
}

/// Wall-clock comparison of the two contour implementations on benzene.
#[test]
#[ignore]
fn bench_contour() {
    use std::time::Instant;
    let scf = build_with(
        r#"
name = "benzene"
unit = "Angstrom"
position = """
    C   1.39600000   0.00000000   0.00000000
    C   0.69800000   1.20900000   0.00000000
    C  -0.69800000   1.20900000   0.00000000
    C  -1.39600000   0.00000000   0.00000000
    C  -0.69800000  -1.20900000   0.00000000
    C   0.69800000  -1.20900000   0.00000000
    H   2.47900000   0.00000000   0.00000000
    H   1.24000000   2.14700000   0.00000000
    H  -1.24000000   2.14700000   0.00000000
    H  -2.47900000   0.00000000   0.00000000
    H  -1.24000000  -2.14700000   0.00000000
    H   1.24000000  -2.14700000   0.00000000
"""
"#,
        "basis-set-pool/cc-pvdz-rifit",
        qp_default(),
    );
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let plan = GwAoPlan::build(&scf).unwrap();
    let mo_ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    let qp_g = scf.eigenvalues[0].clone();
    let qp_w = qp_g.clone();
    println!("n_bas={} n_aux={} n_o={} n_v={}", scf.mol.num_basis, plan.dims.n_aux, n_o, n_v);

    for n in [n_o - 1, n_o, n_o + 3] {
        let mo_row = pyrest::ri_gw::compute_ri3mo_row(&scf, n);
        let ao_row = tensor_ao::ri_row(&scf, &plan, start_mo + n, start_mo, plan.dims.n_mo);
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            let _ = pyrest::ri_gw::contour_rayon(
                qp_g[n], n, &qp_g, &qp_w, n_o, n_v, num_state, &mo_ri_ov, &mo_row, 1e-6, 0.0,
            );
        }
        let mo = t.elapsed().as_secs_f64() / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            let _ = pyrest::ri_gw::contour_rayon_ao(
                &scf, &plan, qp_g[n], n, &qp_g, &qp_w, n_o, n_v, num_state, &ao_row, 1e-6, 0.0,
            );
        }
        let ao = t.elapsed().as_secs_f64() / reps as f64;
        println!("n={:3}: contour MO {:8.2} ms | AO {:8.2} ms | ratio {:.2}", n, mo * 1e3, ao * 1e3, ao / mo);

        // isolate the streaming response build
        let reps2 = 5;
        let t = Instant::now();
        for _ in 0..reps2 {
            let _ = pyrest::ri_gw::response_matrix(&qp_w, n_o, n_v, &mo_ri_ov, 0.05, 'C', 0.0);
        }
        let rm = t.elapsed().as_secs_f64() / reps2 as f64;
        let t = Instant::now();
        for _ in 0..reps2 {
            let _ = tensor_ao::response_matrix(&scf, &plan, &qp_w, 0.05, 'C', 0.0);
        }
        let ra = t.elapsed().as_secs_f64() / reps2 as f64;
        println!("      response MO {:8.2} ms | AO {:8.2} ms | ratio {:.2}", rm * 1e3, ra * 1e3, ra / rm);
    }
}
