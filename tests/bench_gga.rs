//! Micro-benchmark of the GGA grid kernel (`gga_grad_sum`) used by both the
//! ground-state and the TDDFT response gradient.
//!
//! Opt-in: `REST_BENCH_GGA=1 cargo test --release -p rest --test bench_gga -- --nocapture`

use pyrest::ctrl_io;
use pyrest::dft::num_int::eval_ao_batch;
use pyrest::grad::rks::gga_grad_sum;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, SCF};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::MatrixFull;
use rstsr::prelude::*;
use std::time::Instant;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;

const GEOM: &str =
    "C -1.462772 1.122980 0.0\nC -0.733500 0.0 0.0\nC 0.733500 0.0 0.0\n\
     C 1.462772 -1.122980 0.0\nH -0.969284 2.091504 0.0\nH -2.548282 1.066091 0.0\n\
     H -1.252172 -0.955274 0.0\nH 1.252172 0.955274 0.0\nH 0.969284 -2.091504 0.0\n\
     H 2.548282 -1.066091 0.0";

#[test]
fn bench_gga_kernel() {
    if std::env::var("REST_BENCH_GGA").is_err() {
        return;
    }
    // Build geometry + grid without running SCF (grids are geometry-only).
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
    max_scf_cycle = 2

[geom]
    name = "bench"
    unit = "Angstrom"
    position = """
{GEOM}
    """
"##
    );
    let keys = toml::from_str::<serde_json::Value>(&token[..]).unwrap();
    let (ctrl, geom_parsed) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom_parsed, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);

    let nao = scf.mol.num_basis;
    let ng = scf.grids.as_ref().unwrap().weights.len();
    let coords = &scf.grids.as_ref().unwrap().coordinates;
    println!("[bench] nao={} ngrids={}", nao, ng);

    let t0 = Instant::now();
    let ao = eval_ao_batch(&scf.mol, coords, 2, ng);
    println!("[bench] eval_ao_batch(deriv=2) = {:.3}s", t0.elapsed().as_secs_f64());

    let device = DeviceBLAS::default();
    let ao_ten = rt::asarray((&ao.data, [nao, ng, 10].f(), &device));
    let wv: Vec<f64> = (0..ng * 4).map(|i| ((i * 37 % 101) as f64) * 1e-3).collect();
    let wv_ten = rt::asarray((&wv, [ng, 4].f(), &device));

    // warm-up
    let mut v: Tsr<f64> = rt::zeros(([nao, nao, 3].f(), &device));
    gga_grad_sum(&mut v, ao_ten.view(), wv_ten.view());
    let _ = v.sum();

    const N: usize = 5;
    let t0 = Instant::now();
    for _ in 0..N {
        let mut v: Tsr<f64> = rt::zeros(([nao, nao, 3].f(), &device));
        gga_grad_sum(&mut v, ao_ten.view(), wv_ten.view());
    }
    let t_gga = t0.elapsed().as_secs_f64() / N as f64;
    println!("[bench] gga_grad_sum            = {:.3}s/call", t_gga);

    // Reference: the same 15 matmuls expressed as explicit OpenBLAS dgemm calls
    // on contiguous MatrixFull operands (what PySCF's numpy path effectively does).
    let a = |c: usize| -> MatrixFull<f64> {
        MatrixFull::from_vec([nao, ng], ao.data[c * nao * ng..(c + 1) * nao * ng].to_vec()).unwrap()
    };
    let mut aow: Vec<MatrixFull<f64>> = Vec::new();
    for c in 0..4 {
        let mut m = a(c);
        for (g, w) in wv[c * ng..(c + 1) * ng].iter().enumerate() {
            for mu in 0..nao {
                m.data[mu + g * nao] *= w;
            }
        }
        aow.push(m);
    }
    let t0 = Instant::now();
    for _ in 0..N {
        let mut vmat: Vec<MatrixFull<f64>> =
            (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();
        for ic in 0..4 {
            for d in 0..3 {
                let am = a(1 + d);
                let mut tmp = MatrixFull::new([nao, nao], 0.0);
                _dgemm_full(&am, 'N', &aow[ic], 'T', &mut tmp, 1.0, 0.0);
                for k in 0..nao * nao {
                    vmat[d].data[k] += tmp.data[k];
                }
            }
        }
        let _ = &vmat;
    }
    let t_ref = t0.elapsed().as_secs_f64() / N as f64;
    println!("[bench] explicit dgemm (12)      = {:.3}s/call", t_ref);

    let t0 = Instant::now();
    for _ in 0..N {
        let mut vmat: Vec<MatrixFull<f64>> =
            (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();
        for d in 0..3 {
            for ic in 0..4 {
                let am = a(1 + d);
                _dgemm_full(&am, 'N', &aow[ic], 'T', &mut vmat[d], 1.0, 1.0);
            }
        }
        let _ = &vmat;
    }
    let t_ref2 = t0.elapsed().as_secs_f64() / N as f64;
    println!("[bench] explicit dgemm (acc)     = {:.3}s/call", t_ref2);
}
