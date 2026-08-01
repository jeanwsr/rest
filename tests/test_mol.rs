use pyrest::basis_io::{BasCell, Basis4Elem};
use pyrest::constants::{ATM_NUC, BAS_ANG, BAS_CTR, BAS_PRM, ENV_PRT_START};
use pyrest::geom_io::{GeomCell, GeomUnit};
use pyrest::molecule_io::{build_cint, sanity_check_nelec};
use rest_libcint::CintType;
use rest_tensors::MatrixFull;

#[test]
fn test_sanity_check_nelec() {
    let (n, s) = sanity_check_nelec(10.0, 1.0);
    assert_eq!(n, 10);
    assert_eq!(s, 1);
}

#[test]
#[should_panic]
fn test_sanity_check_nelec_panic1() {
    let (_, _) = sanity_check_nelec(10.1, 1.0);
}

#[test]
#[should_panic]
fn test_sanity_check_nelec_panic2() {
    let (_, _) = sanity_check_nelec(10.0, 0.0);
}

#[test]
fn test_single_h_sto3g_spheric() {
    let basis_per_atom = vec![Basis4Elem {
        electron_shells: vec![BasCell {
            function_type: None,
            region: None,
            angular_momentum: vec![0],
            exponents: vec![3.42525091, 0.62391373, 0.16885540],
            coefficients: vec![vec![0.15432897, 0.53532814, 0.44463454]],
            native_coefficients: vec![],
        }],
        references: None,
        ecp_potentials: None,
        ecp_electrons: None,
        global_index: (0, 1),
    }];

    let geom = GeomCell {
        name: "H".into(),
        elem: vec!["H".into()],
        fix: vec![false],
        unit: GeomUnit::Angstrom,
        position: MatrixFull::from_vec([3, 1], vec![0.0, 0.0, 0.0]).unwrap(),
        nfree: 1,
        ..GeomCell::init_geom()
    };

    let (atm, bas, env, _bas_info, _cint_fdqc, num_elec, nbasis, nstate, ecpbas) =
        build_cint(&basis_per_atom, &geom, &CintType::Spheric, 0.0, 2.0, true);

    assert_eq!(nbasis, 1, "H STO-3G spheric -> 1 basis function");
    assert_eq!(nstate, 1);
    assert_eq!(atm.len(), 1);
    assert_eq!(atm[0][ATM_NUC] as i32, 1);
    assert_eq!(bas.len(), 1);
    assert_eq!(bas[0][BAS_ANG] as usize, 0);
    assert_eq!(bas[0][BAS_PRM] as usize, 3);
    assert_eq!(bas[0][BAS_CTR] as usize, 1);
    assert_eq!(num_elec[0], 1.0);
    assert_eq!(num_elec[1], 1.0);
    assert_eq!(num_elec[2], 0.0);
    assert!(ecpbas.is_none());

    let ncoord = 4;
    let nbas_env = 3 + 3;
    assert_eq!(env.len(), ENV_PRT_START + ncoord + nbas_env);
    assert_eq!(env[ENV_PRT_START], 0.0);
    assert_eq!(env[ENV_PRT_START + 1], 0.0);
    assert_eq!(env[ENV_PRT_START + 2], 0.0);
    assert_eq!(env[ENV_PRT_START + 3], 0.0);
    assert!((env[ENV_PRT_START + 4] - 3.42525091).abs() < 1e-8);
}

#[test]
fn test_h2_two_atom() {
    let s_shell = BasCell {
        function_type: None,
        region: None,
        angular_momentum: vec![0],
        exponents: vec![1.0],
        coefficients: vec![vec![1.0]],
        native_coefficients: vec![],
    };
    let basis_per_atom = vec![
        Basis4Elem {
            electron_shells: vec![s_shell.clone()],
            references: None,
            ecp_potentials: None,
            ecp_electrons: None,
            global_index: (0, 1),
        },
        Basis4Elem {
            electron_shells: vec![s_shell.clone()],
            references: None,
            ecp_potentials: None,
            ecp_electrons: None,
            global_index: (1, 1),
        },
    ];

    let geom = GeomCell {
        name: "H2".into(),
        elem: vec!["H".into(), "H".into()],
        fix: vec![false, false],
        unit: GeomUnit::Angstrom,
        position: MatrixFull::from_vec([3, 2], vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.4]).unwrap(),
        nfree: 2,
        ..GeomCell::init_geom()
    };

    let (atm, bas, env, _bas_info, _cint_fdqc, num_elec, nbasis, _nstate, ecpbas) =
        build_cint(&basis_per_atom, &geom, &CintType::Spheric, 0.0, 1.0, true);

    assert_eq!(nbasis, 2);
    assert_eq!(atm.len(), 2);
    assert_eq!(atm[0][ATM_NUC] as i32, 1);
    assert_eq!(atm[1][ATM_NUC] as i32, 1);
    assert_eq!(bas.len(), 2);
    assert_eq!(num_elec[0], 2.0);
    assert!(ecpbas.is_none());

    let ncoord = 4 * 2;
    let nbas_env = 2 * (1 + 1);
    assert_eq!(env.len(), ENV_PRT_START + ncoord + nbas_env);
    assert!((env[ENV_PRT_START + 4 + 2] - 1.4).abs() < 1e-8);
}
