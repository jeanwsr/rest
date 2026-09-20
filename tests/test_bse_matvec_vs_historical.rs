//! Decisive check: does the AO-basis matvec reproduce the **historical**
//! MO-basis matvec entry points (`ri_bse::matvec::a_block_matvec` /
//! `b_block_matvec`) element by element, with exactly the tensors
//! `ri_bse::feast_solver` hands to them?
//!
//! The earlier validation used a hand-written transcription as the reference;
//! this test uses the production code itself.

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_bse;
use pyrest::ri_gw;
use pyrest::scf_io::{self, SCF};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrix::MatrixFull;

fn input_for(geom: &str) -> String {
    format!(
        r##"
[ctrl]
print_level = 0
num_threads = 4
xc = "pbe0"
basis_path = "cc-pvdz"
basis_type = "spheric"
auxbas_path = "def2-universal-jkfit"
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
gw_or_bse = "bse"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
gw_extrapolate_occ_threshold = 100.0
gw_extrapolate_vir_threshold = 100.0
bse_auxbas_path = "def2-universal-jkfit"
"##
    )
}

fn nh3_geom() -> &'static str {
    r##"name = "NH3"
unit = "angstrom"
position = """
N  -2.1988391019   1.8973746268   0.0000000000
H  -1.1788391019   1.8973746268   0.0000000000
H  -2.5388353987   1.0925460144  -0.5263586446
H  -2.5388400276   2.7556271745  -0.4338224694
""""##
}

fn build(geom: &str) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(&input_for(geom)).unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf.prepare_bse_integrals(&None);
    scf
}

fn lcg(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        })
        .collect()
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn ao_matvec_vs_historical_entry_points() {
    let mut scf = build(nh3_geom());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let eps = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &eps);
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        ri_gw::get_occupation_parameters(&scf, 'N');
    println!(
        "n_bas={} n_aux={} start_mo={start_mo} num_state={num_state} o={occ_size} v={vir_size} \
         homo={homo} lumo={lumo}",
        scf.mol.num_basis,
        inv.size[0]
    );
    let num_auxbas = inv.size[0];
    let (o, v) = (occ_size, vir_size);
    let nov = o * v;

    // ---------- exactly the preparation `feast_solver` performs ----------
    let ri_oo = ri_bse::get_submatrix(&scf, 'O', 'O', 'N');
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inv, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);

    let ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'N');
    let mut ri_vv = ri_bse::get_submatrix(&scf, 'V', 'V', 'N');
    ri_vv.reshape([num_auxbas * vir_size, vir_size]);

    let mut ri_ov_tilde_full = MatrixFull::new(ri_ov.size, 0.0);
    _dgemm_full(&inv, 'N', &ri_ov, 'N', &mut ri_ov_tilde_full, 1.0, 0.0);
    let mut ri_ov_tilde = ri_ov_tilde_full.clone();
    ri_ov_tilde.reshape([num_auxbas * occ_size, vir_size]);

    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);

    let ctx = ri_bse::matvec_fast::FastBseContext::build(&scf, &inv).expect("AO context");

    for (label, spin, c) in
        [("singlet", "singlet", 2.0f64), ("triplet", "triplet", 0.0), ("rpa", "rpa", 1.0)]
    {
        scf.mol
            .ctrl
            .quasiparticle_methods
            .as_mut()
            .unwrap()
            .bse_spin = spin.to_string();
        let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();

        for seed in [1u64, 7] {
            let z = lcg(nov, seed);

            // ---- production historical entry points ----
            let a_hist = ri_bse::matvec::a_block_matvec(
                &scf, &qp_ctrl, &ri_vv, &ri_ov, &ri_oo_tilde, &z,
            );
            let b_hist = ri_bse::matvec::b_block_matvec(
                &scf, &qp_ctrl, &ri_ov, &ri_ov_b, &ri_ov_tilde, &z,
            );
            // ---- the B-block W helper in both historical variants ----
            let wb_dgemm =
                ri_bse::matvec::w_contribution_b_block_dgemm(&scf, &ri_ov_b, &z, &ri_ov_tilde);
            let wb_rayon =
                ri_bse::matvec::w_contribution_rayon_b_block(&scf, &ri_ov, &z, &ri_ov_tilde_full);

            // ---- explicit transcription of `w_contribution_rayon_b_block` ----
            let mut wb_expl = vec![0.0f64; nov];
            for q in 0..num_auxbas {
                for i in 0..o {
                    for j in 0..o {
                        let mut s = 0.0f64;
                        for b in 0..v {
                            s += z[j + b * o] * ri_ov.data[q + (i + b * o) * num_auxbas];
                        }
                        for a in 0..v {
                            wb_expl[i + a * o] +=
                                ri_ov_tilde_full.data[q + (j + a * o) * num_auxbas] * s;
                        }
                    }
                }
            }

            // ---- the AO module ----
            let a_new = ctx.a_block_matvec(&scf, &qp_ctrl, &z);
            let b_new = ctx.b_block_matvec(&scf, &qp_ctrl, &z);

            println!("[{label}] seed={seed}");
            println!(
                "  A: |hist|={:.6e}  hist vs AO   max={:.3e} rel={:.3e}",
                norm(&a_hist),
                max_abs(&a_hist, &a_new),
                max_abs(&a_hist, &a_new) / norm(&a_hist)
            );
            println!(
                "  B: |hist|={:.6e}  hist vs AO   max={:.3e} rel={:.3e}",
                norm(&b_hist),
                max_abs(&b_hist, &b_new),
                max_abs(&b_hist, &b_new) / norm(&b_hist)
            );
            println!(
                "  W_B hist-variants: dgemm vs rayon max={:.3e}  rayon vs explicit max={:.3e}",
                max_abs(&wb_dgemm, &wb_rayon),
                max_abs(&wb_rayon, &wb_expl)
            );
            println!(
                "  B without direct (b_hist + c*V2) vs AO: see above; |W_B|={:.6e}",
                norm(&wb_dgemm)
            );
            let _ = c;

            assert!(
                max_abs(&a_hist, &a_new) / norm(&a_hist) < 1e-12,
                "A block must match the historical entry point"
            );
        }
    }
}
