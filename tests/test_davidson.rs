use pyrest::solvers::davidson::{davidson_solver, DavidsonConfig, generate_initial_guess};
use rest_tensors::matrix::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::_dsyev;

#[test]
fn test_davidson_vs_dsyev() {
    let n = 10;
    let mut a = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        a[[i, i]] = (i + 1) as f64 * 0.5;
    }
    for i in 0..n - 1 {
        a[[i, i + 1]] = 0.2;
        a[[i + 1, i]] = 0.2;
    }

    let nroots = 3;
    let (_, evals, _) = _dsyev(&a, 'V');
    let ref_evals = &evals[..nroots];

    let diag: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    let a_matvec = |v: &Vec<f64>| -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| a[[i, j]] * v[j]).sum())
            .collect()
    };
    let initial_guess = generate_initial_guess(&diag, nroots);
    let config = DavidsonConfig { add_dim: 4, ..DavidsonConfig::default() };
    let eigenpairs = davidson_solver(a_matvec, nroots, &diag, initial_guess, &config);

    assert_eq!(eigenpairs.len(), nroots);
    for (i, (e_dav, _v)) in eigenpairs.iter().enumerate() {
        assert!(
            (e_dav - ref_evals[i]).abs() < 1e-8,
            "root {}: davidson={}, dsyev={}",
            i,
            e_dav,
            ref_evals[i]
        );
    }
}
