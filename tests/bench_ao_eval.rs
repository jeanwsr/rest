//! Micro-benchmark + consistency check of the grid AO tables:
//!   * legacy `eval_ao_batch` (used by the TDDFT gradient),
//!   * libcint `CInt::eval_gto` (used by `NIMatmul` in the AO-mode TDDFT driver),
//!   * the SCF's own `grids.ao` / `grids.aop` tables.
//!
//! Opt-in: `REST_BENCH_AOEVAL=1 cargo test --release --test bench_ao_eval -- --nocapture`

use pyrest::ctrl_io;
use pyrest::dft::num_int::eval_ao_batch;
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::scf_io::{self, SCF};
use std::time::Instant;

const C10H8: &str = "C -1.2124 1.4000 0.0\nC -2.4248 0.7000 0.0\nC -2.4248 -0.7000 0.0\n\
C -1.2124 -1.4000 0.0\nC 0.0000 -0.7000 0.0\nC 0.0000 0.7000 0.0\n\
C 1.2124 1.4000 0.0\nC 2.4248 0.7000 0.0\nC 2.4248 -0.7000 0.0\n\
C 1.2124 -1.4000 0.0\nH -1.2124 2.4800 0.0\nH -3.3601 1.2400 0.0\n\
H -3.3601 -1.2400 0.0\nH -1.2124 -2.4800 0.0\nH 1.2124 2.4800 0.0\n\
H 3.3601 1.2400 0.0\nH 3.3601 -1.2400 0.0\nH 1.2124 -2.4800 0.0";

fn build(geom: &str) -> SCF {
    let token = format!(
        r##"
[ctrl]
    print_level = 0
    num_threads = 1
    xc = "blyp"
    basis_path = "def2-svp"
    auxbas_path = "def2-svp-rifit"
    eri_type = "ri-v"
    charge = 0.0
    spin = 1.0
    spin_polarization = false
    initial_guess = "hcore"
    max_scf_cycle = 100
    scf_acc_rho = 1.0e-10
    scf_acc_eev = 1.0e-9
    scf_acc_etot = 1.0e-12

[tddft]
    tddft_method = "tda"
    tddft_spin = "singlet"
    nroots = 3
    davidson_tol = 1.0e-8
    davidson_max_iter = 80

[geom]
    name = "bench"
    unit = "Angstrom"
    position = """
{geom}
    """
"##
    );
    let keys = toml::from_str::<serde_json::Value>(&token[..]).unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    SCF::build(mol, &None)
}

#[test]
fn bench_ao_eval() {
    if std::env::var("REST_BENCH_AOEVAL").is_err() {
        return;
    }
    let mut scf = build(C10H8);
    scf_io::scf_without_build(&mut scf, &None);
    let grids = scf.grids.as_ref().unwrap();
    let ng = grids.weights.len();
    let nao = scf.mol.num_basis;
    println!("nao={} ngrids={}", nao, ng);

    for deriv in [1usize, 2] {
        let t = Instant::now();
        let ao = eval_ao_batch(&scf.mol, &grids.coordinates, deriv, ng);
        let dt = t.elapsed().as_secs_f64();
        let comp = (deriv + 1) * (deriv + 2) * (deriv + 3) / 6;
        println!(
            "[legacy eval_ao_batch deriv={}] {:.3}s  size={} MB",
            deriv,
            dt,
            nao * ng * comp * 8 / 1_000_000
        );
        drop(ao);
    }

    let cint = get_cint_mol(&scf.mol);
    let mut lib: Vec<Vec<f64>> = Vec::new();
    for name in ["deriv0", "deriv1", "deriv2"] {
        let t = Instant::now();
        let out = cint.eval_gto(name, &grids.coordinates);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "[libcint eval_gto {}] {:.3}s shape={:?} size={} MB",
            name,
            dt,
            out.shape,
            out.out.as_ref().map(|v| v.len()).unwrap_or(0) * 8 / 1_000_000
        );
        lib.push(out.out.unwrap());
    }

    // ---- cost of the production wrapper (libcint + transpose) ------------
    for deriv in [1usize, 2] {
        let t = Instant::now();
        let out = cint.eval_gto(&format!("deriv{}", deriv), &grids.coordinates);
        let raw_t = t.elapsed().as_secs_f64();
        drop(out);
        let t = Instant::now();
        let ao = pyrest::dft::num_int::eval_ao_batch_libcint(&cint, nao, &grids.coordinates, deriv, ng);
        let wrap_t = t.elapsed().as_secs_f64();
        drop(ao);
        println!(
            "[cost] deriv={} libcint={:.3}s wrapper(transposed)={:.3}s -> transpose overhead {:.3}s",
            deriv, raw_t, wrap_t, wrap_t - raw_t
        );
    }

    // ---- layout + value consistency through the production wrapper -------
    for deriv in [0usize, 1, 2] {
        let leg = eval_ao_batch(&scf.mol, &grids.coordinates, deriv, ng);
        let fast = pyrest::dft::num_int::eval_ao_batch_libcint(&cint, nao, &grids.coordinates, deriv, ng);
        assert_eq!(leg.size, fast.size);
        let mut md = 0.0f64;
        for k in 0..leg.data.len() {
            md = md.max((leg.data[k] - fast.data[k]).abs());
        }
        println!(
            "[verify] deriv={} legacy vs libcint(transposed): max|d| = {:.3e} (len {})",
            deriv, md, leg.data.len()
        );
        drop(leg);
        drop(fast);
    }

    println!(
        "[grids] ao={} aop={} ao_compressed={} aop_compressed={} cutoff={} drop_dense={}",
        grids.ao.is_some(),
        grids.aop.is_some(),
        grids.ao_compressed.is_some(),
        grids.aop_compressed.is_some(),
        grids.ao_cutoff,
        scf.mol.ctrl.drop_dense_ao
    );
}
