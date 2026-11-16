//! Grid-source regression tests without basis downloads, AO tabulation, or SCF.

use pyrest::basis_io::{BasCell, Basis4Elem};
use pyrest::dft::{gen_grids, Grids};
use pyrest::molecule_io::Molecule;
use rayon::ThreadPoolBuilder;
use rest_tensors::MatrixFull;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

const PRUNING: [&str; 3] = ["none", "nwchem", "sg1"];

// Frozen 14-point coordinates from the pre-generated-grid table, in its
// original order. The unequal last bits and the signs are intentional.
const LEGACY_14: [[f64; 3]; 14] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.5773502691896258, 0.5773502691896257, 0.5773502691896258],
    [0.5773502691896258, 0.5773502691896257, -0.5773502691896257],
    [0.5773502691896258, -0.5773502691896257, 0.5773502691896258],
    [0.5773502691896258, -0.5773502691896257, -0.5773502691896257],
    [-0.5773502691896257, 0.5773502691896258, 0.5773502691896258],
    [-0.5773502691896257, 0.5773502691896258, -0.5773502691896257],
    [-0.5773502691896257, -0.5773502691896258, 0.5773502691896258],
    [
        -0.5773502691896257,
        -0.5773502691896258,
        -0.5773502691896257,
    ],
];

fn hydrogen(pruning: &str, use_isdf: bool) -> Molecule {
    // Grid construction only uses the geometry and exponent bounds. No
    // Molecule::build_native, basis files, integral engine, or MPI is needed.
    let mut mol = Molecule::init_mol();
    mol.ctrl.use_isdf = use_isdf;
    mol.ctrl.pruning = pruning.to_owned();
    mol.ctrl.rad_grid_method = "treutler".to_owned();
    mol.ctrl.grid_gen_level = 0;
    mol.ctrl.min_num_angular_points = 14;
    mol.ctrl.max_num_angular_points = 14;
    mol.ctrl.external_grids = "none".to_owned();
    mol.geom.elem = vec!["H".to_owned()];
    mol.geom.rg_elem = mol.geom.elem.clone();
    mol.geom.position = MatrixFull::from_vec([3, 1], vec![0.0; 3]).unwrap();
    mol.geom.rg_position = mol.geom.position.clone();
    mol.basis4elem = vec![Basis4Elem {
        electron_shells: vec![BasCell {
            function_type: None,
            region: None,
            angular_momentum: vec![0],
            exponents: vec![1.0],
            coefficients: vec![vec![1.0]],
            native_coefficients: vec![vec![1.0]],
        }],
        references: None,
        ecp_potentials: None,
        ecp_electrons: None,
        global_index: (0, 1),
    }];
    mol
}

#[derive(Debug, PartialEq, Eq)]
struct GridBits {
    coordinates: Vec<[u64; 3]>,
    weights: Vec<u64>,
    quadrature_weights: Vec<u64>,
    atm_idx: Vec<usize>,
}

fn bits(grid: &Grids) -> GridBits {
    GridBits {
        coordinates: grid
            .coordinates
            .iter()
            .map(|r| r.map(f64::to_bits))
            .collect(),
        weights: grid.weights.iter().map(|w| w.to_bits()).collect(),
        quadrature_weights: grid
            .quadrature_weights
            .iter()
            .map(|w| w.to_bits())
            .collect(),
        atm_idx: grid.atm_idx.clone(),
    }
}

fn assert_same_grid(actual: &Grids, expected: &Grids) {
    assert!(
        bits(actual) == bits(expected),
        "grid coordinates, weights, or atom metadata changed"
    );
}

fn shell_ranges(grid: &Grids) -> Vec<std::ops::Range<usize>> {
    // Every rule begins with its sole +x-axis point. This identifies radial
    // shells without reproducing the radial formula or pruning algorithm.
    let starts: Vec<usize> = grid
        .coordinates
        .iter()
        .enumerate()
        .filter(|(_, r)| r[0] > 0.0 && r[1] == 0.0 && r[2] == 0.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(starts.first(), Some(&0));
    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| start..starts.get(i + 1).copied().unwrap_or(grid.coordinates.len()))
        .collect()
}

fn assert_source(grid: &Grids, use_isdf: bool) {
    assert_eq!(grid.coordinates.len(), grid.weights.len());
    assert_eq!(grid.coordinates.len(), grid.quadrature_weights.len());
    assert!(grid.atm_idx.iter().all(|&i| i == 0));
    assert!(grid
        .weights
        .iter()
        .zip(&grid.quadrature_weights)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    let mut discriminating_shells = 0;
    for range in shell_ranges(grid) {
        let radius = grid.coordinates[range.start][0];
        if use_isdf {
            if range.len() > 6 {
                // For the 14/38/50/74/86/110/194-point rules used here the old
                // table's index 7 has +y/-z. Generated order has +z instead.
                let point = grid.coordinates[range.start + 7];
                assert!(
                    point[0] >= 0.0 && point[1] > 0.0 && point[2] < 0.0,
                    "{}-point shell did not retain historical ordering",
                    range.len()
                );
                discriminating_shells += 1;
            }
        } else {
            let (angular, _) = gen_grids::angular_grid(range.len());
            for (i, &(x, y, z)) in angular.iter().enumerate() {
                let expected = [0.0 + radius * x, 0.0 + radius * y, 0.0 + radius * z];
                assert_eq!(
                    grid.coordinates[range.start + i].map(f64::to_bits),
                    expected.map(f64::to_bits),
                    "generated grid mismatch at angular index {i}"
                );
            }
        }
    }
    if use_isdf {
        assert!(
            discriminating_shells > 0,
            "fixture did not distinguish grid sources"
        );
    }
}

#[test]
fn public_atom_grid_keeps_generated_source_for_every_pruning_mode() {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            for pruning in PRUNING {
                let mol = hydrogen(pruning, false);
                let (coordinates, weights, quadrature) = gen_grids::atom_grid(
                    HashMap::from([(0, 1.0)]),
                    1.0,
                    mol.ctrl.radial_precision,
                    14,
                    14,
                    vec![1],
                    0,
                    vec![(0.0, 0.0, 0.0)],
                    mol.ctrl.hardness,
                    pruning.to_owned(),
                    "treutler".to_owned(),
                    0,
                    gen_grids::RadiiAdjust::Becke,
                );
                let grid = Grids::build_with_level(&mol, 0);
                assert_source(&grid, false);
                let atom_coordinates: Vec<[u64; 3]> = coordinates
                    .iter()
                    .map(|&(x, y, z)| [x.to_bits(), y.to_bits(), z.to_bits()])
                    .collect();
                assert!(atom_coordinates == bits(&grid).coordinates);
                assert!(
                    weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>() == bits(&grid).weights
                );
                assert!(
                    quadrature.iter().map(|w| w.to_bits()).collect::<Vec<_>>()
                        == bits(&grid).quadrature_weights
                );
            }
        });
}

#[test]
fn molecular_builders_route_both_sources_and_explicit_levels() {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            for pruning in PRUNING {
                for use_isdf in [false, true] {
                    let mut mol = hydrogen(pruning, use_isdf);
                    let ordinary = Grids::build(&mut mol);
                    let same_level = Grids::build_with_level(&mol, 0);
                    assert_same_grid(&ordinary, &same_level);
                    assert_source(&ordinary, use_isdf);

                    let finer = Grids::build_with_level(&mol, 1);
                    assert_source(&finer, use_isdf);
                    assert_eq!(shell_ranges(&ordinary).len(), 10);
                    assert_eq!(shell_ranges(&finer).len(), 30);
                    assert_ne!(ordinary.coordinates.len(), finer.coordinates.len());
                }
            }
        });
}

#[test]
fn isdf_retains_historical_coordinates_and_angular_weight_ratios() {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let mut mol = hydrogen("none", true);
            let legacy = Grids::build(&mut mol);
            let generated = Grids::build_with_level(&hydrogen("none", false), 0);
            assert_ne!(bits(&legacy), bits(&generated));
            for range in shell_ranges(&legacy) {
                assert_eq!(range.len(), LEGACY_14.len());
                let radius = legacy.coordinates[range.start][0];
                for (i, point) in LEGACY_14.iter().enumerate() {
                    let expected = point.map(|x| 0.0 + radius * x);
                    assert_eq!(
                        legacy.coordinates[range.start + i].map(f64::to_bits),
                        expected.map(f64::to_bits)
                    );
                }
                // The diagonal weight is 0.075 in both rules; the old rounded axis
                // weight is 0.066666666666667. Their ratio cancels the radial factor.
                let ratio = legacy.weights[range.start] / legacy.weights[range.start + 6];
                let expected = 0.066666666666667_f64 / 0.075_f64;
                assert!(
                    (ratio - expected).abs() <= 4.0 * f64::EPSILON,
                    "historical angular weights were replaced or changed"
                );
            }
        });
}

#[test]
fn interleaved_and_concurrent_molecules_do_not_share_a_grid_mode() {
    let serial = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let (generated, legacy) = serial.install(|| {
        let first = Grids::build(&mut hydrogen("none", false));
        let legacy = Grids::build(&mut hydrogen("none", true));
        let last = Grids::build(&mut hydrogen("none", false));
        assert_same_grid(&first, &last);
        assert_source(&first, false);
        assert_source(&legacy, true);
        (bits(&first), bits(&legacy))
    });
    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = [(false, generated), (true, legacy)]
        .into_iter()
        .map(|(use_isdf, expected)| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                // Separate local pools also avoid changing the process-global
                // Rayon configuration while the Rust test harness is parallel.
                ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
                    .install(|| {
                        barrier.wait();
                        for _ in 0..8 {
                            let mut mol = hydrogen("none", use_isdf);
                            let actual = Grids::build(&mut mol);
                            assert!(
                                bits(&actual) == expected,
                                "concurrent molecule changed grid source"
                            );
                            let level = Grids::build_with_level(&mol, 0);
                            assert!(
                                bits(&level) == expected,
                                "concurrent level grid changed source"
                            );
                        }
                    });
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}

struct ExternalGridFile(PathBuf);

impl ExternalGridFile {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rest-isdf-grid-{}-{}.txt",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        for i in 0..7 {
            // Exact binary fractions, signed weights, and a non-Lebedev size
            // expose any accidental reconstruction or weight normalization.
            let x = i as f64 + 0.25;
            let y = -(i as f64) - 0.5;
            let z = i as f64 + 0.75;
            let w = if i % 2 == 0 { 0.125 } else { -0.25 };
            writeln!(file, "{x:.16e}, {y:.16e}, {z:.16e}, {w:.16e}").unwrap();
        }
        Self(path)
    }
}

impl Drop for ExternalGridFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn external_grids_keep_input_values_for_both_isdf_modes() {
    let external = ExternalGridFile::new();
    for threads in [1, 3] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let order: Vec<usize> = if threads == 1 {
                    (0..7).collect()
                } else {
                    vec![0, 3, 6, 1, 4, 2, 5]
                };
                let mut previous = None;
                for use_isdf in [false, true] {
                    let mut mol = hydrogen("none", use_isdf);
                    mol.ctrl.external_grids = external.0.to_str().unwrap().to_owned();
                    let grid = Grids::build(&mut mol);
                    assert_eq!(grid.coordinates.len(), order.len());
                    assert_eq!(grid.atm_idx, vec![usize::MAX; order.len()]);
                    for (position, &i) in order.iter().enumerate() {
                        let expected = [i as f64 + 0.25, -(i as f64) - 0.5, i as f64 + 0.75];
                        let weight: f64 = if i % 2 == 0 { 0.125 } else { -0.25 };
                        assert_eq!(
                            grid.coordinates[position].map(f64::to_bits),
                            expected.map(f64::to_bits)
                        );
                        assert_eq!(grid.weights[position].to_bits(), weight.to_bits());
                        assert_eq!(
                            grid.quadrature_weights[position].to_bits(),
                            weight.to_bits()
                        );
                    }
                    if let Some(ref expected) = previous {
                        assert!(bits(&grid) == *expected, "ISDF flag changed imported grids");
                    }
                    previous = Some(bits(&grid));
                }
            });
    }
}
