/// CP-HF test: dense solver self-consistency.

// ── helper: build H2O SCF (HF/sto-3g) ──────────────────────────────────────
fn run_h2o_scf_hf_sto3g() -> pyrest::scf_io::SCF {
    use pyrest::ctrl_io;
    use pyrest::molecule_io::Molecule;
    use pyrest::scf_io::{self, SCF};

    let input_token = r##"
[ctrl]
     print_level =          0
     xc =                   "hf"
     basis_path =           "sto-3g"
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

// ── Test: dense solver residual check ──────────────────────────────────────
#[test]
fn test_cphf_dense_solver_residual() {
    use pyrest::ri_cphf::CPHFSolverPySCF;

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolverPySCF::new(&scf);
    println!("\n  === H2O HF/STO-3G SCF ===");
    println!("  HF energy:        {:.12} Ha", scf.scf_energy);
    println!("  nocc = {}, nvir = {}, dim = {}", solver.nocc, solver.nvir, solver.dim);

    let mo_dip = build_full_mo_dipole(&scf);
    let s1_zero = vec![0.0; solver.nmo * solver.nocc];

    // Build h1 in (nmo, nocc) format
    let h1_nmc: Vec<Vec<f64>> = (0..3).map(|comp| {
        let mut h1 = vec![0.0; solver.nmo * solver.nocc];
        for col in 0..solver.nocc {
            let occ_idx = solver.start_mo + col;
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                h1[row + col * solver.nmo] = mo_dip[comp][[occ_idx, row]];
            }
        }
        h1
    }).collect();

    let labels = ["x", "y", "z"];
    for (comp, h1) in h1_nmc.iter().enumerate() {
        let u_full = solver.solve_dense(&scf, None, h1, &s1_zero)
            .expect("dense solve failed");

        // Extract VO block
        let mut u_vo = vec![0.0; solver.dim];
        for col in 0..solver.nocc {
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                u_vo[col + a * solver.nocc] = u_full[row + col * solver.nmo];
            }
        }

        // Build z_full = u_vo expanded to (nmo, nocc), zeros in OO
        let mut z_full = vec![0.0; solver.nmo * solver.nocc];
        for col in 0..solver.nocc {
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                z_full[row + col * solver.nmo] = u_vo[col + a * solver.nocc];
            }
        }

        let fv = solver.fvind_nmo_nocc(&scf, None, &z_full);

        // Residual: r = u_vo + G_tilde(u_vo) + h1_vo * e_ai
        // where G_tilde = fvind_nmo_nocc(..., )[VO_block] * e_ai
        let residual: f64 = (0..solver.dim).map(|ia| {
            let col = ia % solver.nocc;
            let a = ia / solver.nocc;
            let row = solver.lumo + a;
            let g_vo = fv[row + col * solver.nmo];
            let h1_vo = h1[row + col * solver.nmo];
            let r = u_vo[ia] + g_vo * solver.e_ai[ia] + h1_vo * solver.e_ai[ia];
            r * r
        }).sum();

        let rnorm = residual.sqrt();
        println!("  Dense |(I+G)·U + h·e_ai| ({}) = {:.2e}", labels[comp], rnorm);
        assert!(rnorm < 1e-14,
            "Dense solver residual too large for {}: {:.2e}", labels[comp], rnorm);
    }
    println!("  ✓ Dense solver satisfies CP-HF equation to machine precision");
}

/// Helper: build full MO dipole matrix from SCF data.
fn build_full_mo_dipole(scf: &pyrest::scf_io::SCF) -> [rest_tensors::MatrixFull<f64>; 3] {
    use rest_tensors::MatrixFull;
    use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;

    let ao_dip = pyrest::ri_bse::dipoles::obtain_ao_dips(scf, None);
    let eigvec = &scf.eigenvectors[0];
    let nao = eigvec.size[0];
    let nmo = eigvec.size[1];

    let mut result = [(); 3].map(|_| MatrixFull::new([nmo, nmo], 0.0));

    for comp in 0..3 {
        let ao_dip_comp = ao_dip.get_reducing_matrix(comp).unwrap();
        let mut tmp = MatrixFull::new([nao, nmo], 0.0);
        _dgemm_full(eigvec, 'T', &ao_dip_comp, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', eigvec, 'N', &mut result[comp], 1.0, 0.0);
    }
    result
}
