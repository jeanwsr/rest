use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, SCF};
use pyrest::dft::num_int::{prepare_fxc_data, fxc_matvec};
use pyrest::dft::num_int::{prepare_fxc_data_nwchem_opt, fxc_matvec_nwchem_opt};
use pyrest::ri_tddft::utils::tddft_occupation_parameters;

fn run_h2o_scf() -> SCF {
    let input_token = r##"
[ctrl]
     print_level =          0
     xc =                   "blyp"
     basis_path =           "def2-svp"
     auxbas_path =          "def2-universal-jkfit"
     eri_type =             "ri-v"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess=         "hcore"
     mixer =                "diis"
     num_threads =          8
     scf_acc_rho =          1.0e-8
     scf_acc_eev =          1.0e-6
     scf_acc_etot =         1.0e-8

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
        O     0.00000000     0.00000000     0.12982363
        H     0.75933475     0.00000000    -0.46621158
        H    -0.75933475     0.00000000    -0.46621158
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data
}

#[test]
fn test_fxc_bench_h2o_real() {
    // ── Run SCF for H2O BLYP/def2-SVP ──
    let scf = run_h2o_scf();

    let (start_mo, num_state, nocc, nvir, _homo, _lumo) =
        tddft_occupation_parameters(&scf);
    let ngrids = scf.grids.as_ref().unwrap().weights.len();
    let nvar = if scf.mol.xc_data.use_density_gradient() { 4 } else { 1 };

    println!("\n  === H2O BLYP/def2-SVP SCF converged ===");
    println!("  Energy:           {:.12} Ha", scf.scf_energy);
    println!("  nocc:             {}", nocc);
    println!("  nvir:             {}", nvir);
    println!("  ngrids:           {}", ngrids);
    println!("  nvar:             {}", nvar);
    println!("  start_mo:         {}", start_mo);
    println!("  total MOs:        {}", num_state);
    println!("  (NWChem ref: medium grid ~9000 pts, nocc=5, nvir=14)");

    // ── Prepare fxc kernel data (REST standard layout) ──
    let fxc_data = prepare_fxc_data(&scf);
    assert_eq!(fxc_data.nocc, nocc);
    assert_eq!(fxc_data.nvir, nvir);
    assert_eq!(fxc_data.ngrids, ngrids);

    // ── Prepare fxc kernel data (nwchem-opt layout) ──
    let mut fxc_data_opt = prepare_fxc_data_nwchem_opt(&scf);

    // ── Benchmark ──
    let n = nocc * nvir;
    let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
    let n_repeat = 20;

    use std::time::Instant;

    // Warm-up
    let _ = fxc_matvec(&fxc_data, &z);
    let _ = fxc_matvec_nwchem_opt(&mut fxc_data_opt, &z);

    // Time REST old (serial)
    let t0 = Instant::now();
    for _ in 0..n_repeat {
        let _ = fxc_matvec(&fxc_data, &z);
    }
    let dt_old = t0.elapsed().as_secs_f64() / n_repeat as f64;

    // Time REST nwchem-opt (8-thread rayon pool)
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let dt_opt = pool.install(|| {
        let t0 = Instant::now();
        for _ in 0..n_repeat {
            let _ = fxc_matvec_nwchem_opt(&mut fxc_data_opt, &z);
        }
        t0.elapsed().as_secs_f64() / n_repeat as f64
    });

    let speedup = dt_old / dt_opt;

    println!("\n  === fxc_matvec benchmark (real H2O, BLYP/def2-SVP, grid) ===");
    println!("  REST old (serial):         {:>9.6} s  (ngrids={})", dt_old, ngrids);
    println!("  REST nwchem-opt (8thr):    {:>9.6} s  (ngrids={})", dt_opt, ngrids);
    println!("  speedup (opt vs old):      {:>7.2}×",    speedup);
    println!();
    println!("  ── Compare with NWChem (same molecule, diff grid) ──");
    println!("  NWChem FXC_TIME avg:       ~0.090 s/call  (ngrids~9000)");
    println!("  REST old     vs NWChem:    {:.1}× faster  (ngrids {} vs ~9000)", 0.090 / dt_old, ngrids);
    println!("  REST nwchem-opt vs NWChem: {:.1}× faster  (ngrids {} vs ~9000)", 0.090 / dt_opt, ngrids);
    println!("  (NWChem: 1 MPI + single-thread internal BLAS. REST: 1 or 8 threads)");
    println!("  (Note: diff grid sizes. Scale: REST at ~9000pts is ~{:.3}s)", dt_old * 9000.0 / ngrids as f64);
}
