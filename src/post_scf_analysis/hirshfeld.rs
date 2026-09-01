use crate::dft::Grids;
use crate::geom_io::get_charge;
use crate::initial_guess::sad::initial_guess_from_sad;
use crate::scf_io::SCF;
use rayon::prelude::*;
use std::time::Instant;
use tensors::matrix_blas_lapack::_dgemm_full;
use tensors::{MathMatrix, MatrixFull};

const PRO_DENSITY_THRESHOLD: f64 = 1.0e-30;
const CONSERVATION_WARNING_THRESHOLD: f64 = 1.0e-4;

/// Hirshfeld (stockholder) populations and charges.
///
/// The pro-molecule is assembled from the same neutral, spherically averaged
/// atomic HF densities used by REST's SAD initial guess. All densities are
/// evaluated in the molecular AO basis on the molecular numerical grid.
#[derive(Clone, Debug)]
pub struct HirshfeldAnalysis {
    /// Electron population assigned to each real atom.
    pub populations: Vec<f64>,
    /// Atomic charge `Z_eff - population` for each real atom.
    pub charges: Vec<f64>,
    /// Molecular electron density integrated over the numerical grid.
    pub integrated_electrons: f64,
    /// Sum of the atom-resolved Hirshfeld populations.
    pub assigned_electrons: f64,
    /// Density not assigned because the pro-molecule density underflowed.
    pub unassigned_electrons: f64,
    /// Electron count evaluated analytically as Tr(P S).
    pub ao_electrons: f64,
    /// Numerical-grid electron count minus Tr(P S).
    pub grid_electron_residual: f64,
    /// Assigned electron count minus the numerical-grid electron count.
    pub partition_residual: f64,
    /// Sum of Hirshfeld charges minus the requested molecular charge.
    pub charge_residual: f64,
    /// Wall-time breakdown for the major analysis phases.
    pub timing: HirshfeldTiming,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HirshfeldTiming {
    pub grid_preparation: f64,
    pub pro_molecule: f64,
    pub density_evaluation: f64,
    pub stockholder_partition: f64,
    pub total: f64,
}

struct HirshfeldAccumulator {
    populations: Vec<f64>,
    pro_atom_densities: Vec<f64>,
    integrated_electrons: f64,
    unassigned_electrons: f64,
}

impl HirshfeldAccumulator {
    fn new(num_atoms: usize) -> Self {
        Self {
            populations: vec![0.0; num_atoms],
            pro_atom_densities: vec![0.0; num_atoms],
            integrated_electrons: 0.0,
            unassigned_electrons: 0.0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.populations
            .iter_mut()
            .zip(other.populations.iter())
            .for_each(|(to, from)| *to += from);
        self.integrated_electrons += other.integrated_electrons;
        self.unassigned_electrons += other.unassigned_electrons;
        self
    }
}

fn total_density_matrix(dms: &[MatrixFull<f64>], spin_channel: usize) -> MatrixFull<f64> {
    let mut total = dms[0].clone();
    if spin_channel == 2 {
        total.self_add(&dms[1]);
    }
    total
}

fn partition_density(
    weighted_density: f64,
    pro_atom_densities: &[f64],
    populations: &mut [f64],
) -> bool {
    let denominator = pro_atom_densities.iter().sum::<f64>();
    if denominator <= PRO_DENSITY_THRESHOLD {
        false
    } else {
        populations
            .iter_mut()
            .zip(pro_atom_densities.iter())
            .for_each(|(population, pro_atom_density)| {
                *population += weighted_density * pro_atom_density / denominator;
            });
        true
    }
}

fn effective_nuclear_charges(scf_data: &SCF) -> Vec<f64> {
    let mut nuclear_charges = get_charge(&scf_data.mol.geom.elem);
    nuclear_charges
        .iter_mut()
        .zip(scf_data.mol.basis4elem.iter())
        .for_each(|(charge, basis)| {
            if let Some(ecp_electrons) = basis.ecp_electrons {
                *charge -= ecp_electrons as f64;
            }
        });
    nuclear_charges
}

enum AnalysisGrids<'a> {
    Borrowed(&'a Grids),
    Owned(Grids),
}

impl AnalysisGrids<'_> {
    fn as_ref(&self) -> &Grids {
        match self {
            Self::Borrowed(grids) => grids,
            Self::Owned(grids) => grids,
        }
    }
}

fn analysis_grids(scf_data: &SCF) -> AnalysisGrids<'_> {
    if let Some(grids) = &scf_data.grids {
        if grids.ao.is_some() {
            return AnalysisGrids::Borrowed(grids);
        }
    }

    let mut grids = if let Some(grids) = &scf_data.grids {
        grids.clone()
    } else {
        let mut molecule = scf_data.mol.clone();
        Grids::build(&mut molecule)
    };

    if grids.ao.is_none() {
        if grids.ao_compressed.is_some() {
            grids.ensure_dense_ao();
        } else {
            grids.prepare_tabulated_ao(&scf_data.mol);
        }
    }
    AnalysisGrids::Owned(grids)
}

/// Evaluate Hirshfeld charges for the converged SCF density.
///
/// Ghost centers contribute basis functions to the molecular density but are
/// not treated as pro-atoms and therefore do not receive a Hirshfeld charge.
pub fn hirshfeld_analysis(scf_data: &SCF) -> HirshfeldAnalysis {
    let total_start = Instant::now();
    let molecule = &scf_data.mol;
    let num_atoms = molecule.geom.elem.len();
    let num_basis = molecule.num_basis;
    let phase_start = Instant::now();
    let analysis_grids = analysis_grids(scf_data);
    let grid_preparation = phase_start.elapsed().as_secs_f64();
    let grids = analysis_grids.as_ref();
    let ao = grids
        .ao
        .as_ref()
        .expect("AO values must be available for Hirshfeld analysis");
    let num_grids = grids.coordinates.len();

    let molecular_dm = total_density_matrix(&scf_data.density_matrix, molecule.spin_channel);
    let phase_start = Instant::now();
    let sad_dms = initial_guess_from_sad(molecule, &None);
    let pro_molecule_dm = total_density_matrix(&sad_dms, molecule.spin_channel);
    let pro_molecule = phase_start.elapsed().as_secs_f64();
    let overlap = scf_data
        .ovlp
        .to_matrixfull()
        .expect("overlap matrix must be available for Hirshfeld analysis");
    let ao_electrons = molecular_dm
        .data
        .iter()
        .zip(overlap.data.iter())
        .map(|(density, overlap)| density * overlap)
        .sum::<f64>();

    let phase_start = Instant::now();
    let mut molecular_dm_ao = MatrixFull::new([num_basis, num_grids], 0.0);
    _dgemm_full(&molecular_dm, 'N', ao, 'N', &mut molecular_dm_ao, 1.0, 0.0);

    let mut pro_molecule_dm_ao = MatrixFull::new([num_basis, num_grids], 0.0);
    _dgemm_full(
        &pro_molecule_dm,
        'N',
        ao,
        'N',
        &mut pro_molecule_dm_ao,
        1.0,
        0.0,
    );
    let density_evaluation = phase_start.elapsed().as_secs_f64();

    let ao_slices = molecule.aoslice_by_atom();
    let phase_start = Instant::now();
    let accumulator = grids
        .weights
        .par_iter()
        .enumerate()
        .fold(
            || HirshfeldAccumulator::new(num_atoms),
            |mut accumulator, (grid_index, grid_weight)| {
                let molecular_density = (0..num_basis)
                    .map(|ao_index| {
                        ao[[ao_index, grid_index]] * molecular_dm_ao[[ao_index, grid_index]]
                    })
                    .sum::<f64>();
                let weighted_density = grid_weight * molecular_density;
                accumulator.integrated_electrons += weighted_density;

                accumulator
                    .pro_atom_densities
                    .iter_mut()
                    .zip(ao_slices[..num_atoms].iter())
                    .for_each(|(pro_atom_density, [_, _, ao_start, ao_end])| {
                        *pro_atom_density = (*ao_start..*ao_end)
                            .map(|ao_index| {
                                ao[[ao_index, grid_index]]
                                    * pro_molecule_dm_ao[[ao_index, grid_index]]
                            })
                            .sum::<f64>()
                            .max(0.0);
                    });

                if !partition_density(
                    weighted_density,
                    &accumulator.pro_atom_densities,
                    &mut accumulator.populations,
                ) {
                    accumulator.unassigned_electrons += weighted_density;
                }
                accumulator
            },
        )
        .reduce(
            || HirshfeldAccumulator::new(num_atoms),
            HirshfeldAccumulator::merge,
        );
    let stockholder_partition = phase_start.elapsed().as_secs_f64();

    let populations = accumulator.populations;
    let integrated_electrons = accumulator.integrated_electrons;
    let unassigned_electrons = accumulator.unassigned_electrons;
    let assigned_electrons = populations.iter().sum::<f64>();
    let charges = effective_nuclear_charges(scf_data)
        .iter()
        .zip(populations.iter())
        .map(|(nuclear_charge, population)| nuclear_charge - population)
        .collect::<Vec<_>>();
    let grid_electron_residual = integrated_electrons - ao_electrons;
    let partition_residual = assigned_electrons - integrated_electrons;
    let charge_residual = charges.iter().sum::<f64>() - scf_data.mol.ctrl.charge;

    HirshfeldAnalysis {
        populations,
        charges,
        integrated_electrons,
        assigned_electrons,
        unassigned_electrons,
        ao_electrons,
        grid_electron_residual,
        partition_residual,
        charge_residual,
        timing: HirshfeldTiming {
            grid_preparation,
            pro_molecule,
            density_evaluation,
            stockholder_partition,
            total: total_start.elapsed().as_secs_f64(),
        },
    }
}

pub fn print_hirshfeld_analysis(scf_data: &SCF) {
    let analysis = hirshfeld_analysis(scf_data);
    println!("Hirshfeld population analysis:");
    println!("  Atom          Population        Charge");
    analysis
        .populations
        .iter()
        .zip(analysis.charges.iter())
        .zip(scf_data.mol.geom.elem.iter())
        .enumerate()
        .for_each(|(atom_index, ((population, charge), element))| {
            println!(
                "{:3}-{:3}: {:14.8}  {:12.8}",
                atom_index, element, population, charge
            );
        });
    println!(
        "  AO electrons Tr(P*S): {:14.8}; grid: {:14.8}; assigned: {:14.8}",
        analysis.ao_electrons, analysis.integrated_electrons, analysis.assigned_electrons
    );
    println!(
        "  Residuals grid-AO: {:10.3e}; partition-grid: {:10.3e}; unassigned: {:10.3e}",
        analysis.grid_electron_residual, analysis.partition_residual, analysis.unassigned_electrons
    );
    println!(
        "  Charge sum: {:14.8}; target molecular charge: {:14.8}",
        analysis.charges.iter().sum::<f64>(),
        scf_data.mol.ctrl.charge
    );
    println!(
        "  Timing [s]: grid={:.4}, pro-atoms={:.4}, density={:.4}, partition={:.4}, total={:.4}",
        analysis.timing.grid_preparation,
        analysis.timing.pro_molecule,
        analysis.timing.density_evaluation,
        analysis.timing.stockholder_partition,
        analysis.timing.total
    );
    if analysis.grid_electron_residual.abs() > CONSERVATION_WARNING_THRESHOLD
        || analysis.partition_residual.abs() > CONSERVATION_WARNING_THRESHOLD
        || analysis.charge_residual.abs() > CONSERVATION_WARNING_THRESHOLD
    {
        println!(
            "  WARNING: Hirshfeld conservation residual exceeds {:.1e}",
            CONSERVATION_WARNING_THRESHOLD
        );
    }
}

#[cfg(test)]
mod tests {
    use super::partition_density;

    #[test]
    fn stockholder_partition_conserves_density() {
        let mut populations = vec![0.0; 3];
        assert!(partition_density(12.0, &[1.0, 2.0, 3.0], &mut populations));
        assert!((populations.iter().sum::<f64>() - 12.0).abs() < 1.0e-14);
        assert!((populations[0] - 2.0).abs() < 1.0e-14);
        assert!((populations[1] - 4.0).abs() < 1.0e-14);
        assert!((populations[2] - 6.0).abs() < 1.0e-14);
    }

    #[test]
    fn zero_pro_molecule_density_is_not_partitioned() {
        let mut populations = vec![0.0; 2];
        assert!(!partition_density(1.0, &[0.0, 0.0], &mut populations));
        assert_eq!(populations, vec![0.0, 0.0]);
    }
}
