use pyrest::basis_io::{BasCell, Basis4Elem};
use pyrest::fileop::chkfile::{geom_from_json, geom_to_json};
use pyrest::geom_io::{GeomCell, GeomUnit};
use pyrest::initial_guess::proj::proj_mo;
use pyrest::molecule_io::{build_cint, Molecule};
use rest_libcint::CintType;
use rest_tensors::MatrixFull;

fn shell4elem(
    angular_momentum: Vec<i32>,
    exponents: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    global_index: usize,
) -> Basis4Elem {
    Basis4Elem {
        electron_shells: vec![BasCell {
            function_type: None, region: None,
            angular_momentum,
            exponents,
            coefficients,
            native_coefficients: vec![],
        }],
        references: None,
        ecp_potentials: None,
        ecp_electrons: None,
        global_index: (global_index, 1),
    }
}

fn geom_h2(r: f64) -> GeomCell {
    GeomCell {
        name: "H2".into(), elem: vec!["H".into(), "H".into()], fix: vec![false, false],
        unit: GeomUnit::Angstrom,
        position: MatrixFull::from_vec([3, 2], vec![0.0, 0.0, 0.0, 0.0, 0.0, r]).unwrap(),
        nfree: 2, ..GeomCell::init_geom()
    }
}

fn h2_basis(exponents: Vec<f64>, coefficients: Vec<Vec<f64>>) -> Vec<Basis4Elem> {
    vec![
        shell4elem(vec![0], exponents.clone(), coefficients.clone(), 0),
        shell4elem(vec![0], exponents, coefficients, 1),
    ]
}

fn make_molecule(basis: &[Basis4Elem], geom: &GeomCell) -> Molecule {
    let (atm, bas, env, fdqc, fdqc_idx, _nb, ecp) =
        build_cint(basis, geom, &CintType::Spheric);
    let mut mol = Molecule::init_mol();
    mol.cint_type = CintType::Spheric;
    mol.set_cint_data(atm, bas, env, ecp, None, None, Some(fdqc), Some(fdqc_idx));
    mol
}

fn identity_mo(nb: usize, ns: usize) -> [MatrixFull<f64>; 2] {
    let mut v = vec![0.0; nb * ns];
    for i in 0..ns.min(nb) {
        v[i * nb + i] = 1.0;
    }
    [MatrixFull::from_vec([nb, ns], v).unwrap(), MatrixFull::empty()]
}

#[test]
fn test_proj_one_electron_h2() {
    let geom = geom_h2(1.4);
    let src_basis = h2_basis(vec![1.0], vec![vec![1.0]]);   // nb=2
    let tgt_basis = vec![
        shell4elem(vec![0], vec![1.0], vec![vec![1.0]], 0),
        shell4elem(vec![0], vec![3.0], vec![vec![0.5]], 0),
        shell4elem(vec![0], vec![1.0], vec![vec![1.0]], 1),
    ];                                                        // nb=3
    let src = make_molecule(&src_basis, &geom);
    let tgt = make_molecule(&tgt_basis, &geom);
    assert_eq!(src.num_basis, 2);
    assert_eq!(tgt.num_basis, 3);

    let mo_src = identity_mo(src.num_basis, src.num_state);
    let mo_tgt = proj_mo(&tgt, &src, mo_src.clone(), [0..1, 0..0]);
    assert_eq!(mo_tgt[0].size, [tgt.num_basis, tgt.num_state],
        "1 col from nb=2 H2+ to nb=3 H2+, padded to ns=3");
}

#[test]
fn test_proj_partial_range_h2_same_geom() {
    let geom = geom_h2(1.4);
    let basis = h2_basis(vec![1.0], vec![vec![1.0]]);
    let src = make_molecule(&basis, &geom);
    let tgt = make_molecule(&basis, &geom);
    assert_eq!(src.num_basis, 2);
    assert_eq!(tgt.num_basis, 2);

    let mo_src = identity_mo(src.num_basis, src.num_state);
    let mo_tgt = proj_mo(&tgt, &src, mo_src.clone(), [0..1, 0..0]);
    assert_eq!(mo_tgt[0].size, [tgt.num_basis, tgt.num_state],
        "1 col projected, padded to 2 cols");
}

#[test]
fn test_proj_diff_geom_h2() {
    let basis = h2_basis(vec![1.0], vec![vec![1.0]]);
    let src = make_molecule(&basis, &geom_h2(1.4));
    let tgt = make_molecule(&basis, &geom_h2(2.0));
    let mo_src = identity_mo(src.num_basis, src.num_state);
    let mo_tgt = proj_mo(&tgt, &src, mo_src.clone(), [0..2, 0..0]);
    assert_eq!(mo_tgt[0].size, [tgt.num_basis, tgt.num_state]);
}

#[test]
fn test_proj_empty_range_returns_empty() {
    let geom = geom_h2(1.4);
    let basis = h2_basis(vec![1.0], vec![vec![1.0]]);
    let src = make_molecule(&basis, &geom);
    let tgt = make_molecule(&basis, &geom);
    let mo_src = identity_mo(src.num_basis, src.num_state);
    let mo_tgt = proj_mo(&tgt, &src, mo_src.clone(), [0..0, 0..0]);
    assert_eq!(mo_tgt[0].size[0], 0, "empty range => empty output");
}

#[test]
fn test_proj_range_exceeds_target_clamped() {
    let geom = geom_h2(1.4);
    let src_basis = vec![
        shell4elem(vec![0], vec![1.0], vec![vec![1.0]], 0),
        shell4elem(vec![0], vec![3.0], vec![vec![0.5]], 0),
        shell4elem(vec![0], vec![1.0], vec![vec![1.0]], 1),
    ];   // nb=3
    let tgt_basis = h2_basis(vec![1.0], vec![vec![1.0]]);  // nb=2
    let src = make_molecule(&src_basis, &geom);
    let tgt = make_molecule(&tgt_basis, &geom);
    assert_eq!(src.num_basis, 3);
    assert_eq!(tgt.num_basis, 2);

    let mo_src = identity_mo(src.num_basis, src.num_state);
    let mo_tgt = proj_mo(&tgt, &src, mo_src.clone(), [0..3, 0..0]);
    assert_eq!(mo_tgt[0].size, [tgt.num_basis, tgt.num_state],
        "range [0..3] clamped to target ns=2");
}

// Geom JSON round-trips ghost atoms (regression: ghosts were dropped, corrupting build_cint).
#[test]
fn test_geom_roundtrip_preserves_ghost_atoms() {
    let geom = GeomCell {
        name: "H+He".into(), elem: vec!["H".into()], fix: vec![false],
        unit: GeomUnit::Bohr,
        position: MatrixFull::from_vec([3, 1], vec![0.0, 0.0, 0.0]).unwrap(),
        nfree: 1,
        ghost_bs_elem: vec!["He".into()],
        ghost_bs_pos: MatrixFull::from_vec([3, 1], vec![0.0, 0.0, 2.0]).unwrap(),
        ..GeomCell::init_geom()
    };

    let json = geom_to_json(&geom);
    let loaded = geom_from_json(&json).unwrap();
    assert_eq!(loaded.elem, geom.elem);
    assert_eq!(loaded.ghost_bs_elem, geom.ghost_bs_elem);
    let pos: Vec<f64> = loaded.ghost_bs_pos.iter().copied().collect();
    let ghost_pos: Vec<f64> = geom.ghost_bs_pos.iter().copied().collect();
    assert_eq!(pos, ghost_pos, "ghost_bs_pos must round-trip");
}

// Old-format chkfile JSON (no ghost keys) still parses, defaulting to no ghosts.
#[test]
fn test_geom_from_json_defaults_to_no_ghosts_for_old_format() {
    let json = serde_json::to_string(&serde_json::json!({
        "name": "H2",
        "elem": ["H", "H"],
        "unit": "angstrom",
        "position": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
    })).unwrap();
    let loaded = geom_from_json(&json).unwrap();
    assert!(loaded.ghost_bs_elem.is_empty(), "old-format chkfile => no ghosts");
    assert_eq!(loaded.elem, vec!["H".to_string(), "H".to_string()]);
}
