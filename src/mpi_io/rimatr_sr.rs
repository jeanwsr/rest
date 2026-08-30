//! MPI-parallel construction of the short-range (SR) decomposed 3-center RI integrals
//! (`rimatr_sr`) for range-separated hybrid (RSH) functionals.
//!
//! ## Background
//!
//! For RSH functionals, the SCF needs — besides the standard full-range `rimatr` — a second
//! decomposed integral tensor built from the short-range Coulomb kernel `erfc(ωr)/r`
//! (see `SCF::prepare_necessary_integrals`, field `SCF::rimatr_sr`). The serial path
//! constructs it in `ri_jk::generate_rimatr_bare` (rstsr-based); under MPI this module
//! provides the equivalent construction, keeping exactly the same data distribution as the
//! full-range MPI `rimatr` (auxiliary-basis/column-distributed, see
//! `MPIData::distribute_rimatr_tasks` and
//! `Molecule::prepare_rimatr_for_ri_v_mpi_rayon`), so the existing MPI J/K kernels
//! (`vk_upper_with_rimatr_use_dm_only_sync_mpi` and friends) apply unchanged.
//!
//! Everything here is implemented on `rest_tensors` (`MatrixFull`) primitives:
//! raw shell-quartet integrals from libcint, `_dgemm_full` for the `J^(-1/2)` solve, and
//! the `mpi_isend_irecv_wrt_distribution_v03` column-block exchange.
//!
//! ## Omega sign convention
//!
//! `omega_libcint` follows libcint's `PTR_RANGE_OMEGA` convention, identical to
//! `ri_jk::generate_rimatr_bare` and `Molecule::prepare_rimatr_for_ri_v_mpi_rayon`:
//! negative values select the short-range kernel `erfc(ωr)/r`. The SCF caller passes
//! `-omega_functional`.

use std::sync::mpsc::channel;

use rayon::prelude::*;
use tensors::matrix_blas_lapack::{
    _dgemm_full, _power_rayon_for_symmetric_matrix, omp_set_num_threads_wrapper,
};
use tensors::{BasicMatrix, MatrixFull};

use crate::constants::AUXBAS_THRESHOLD;
use crate::molecule_io::Molecule;

use super::{mpi_isend_irecv_wrt_distribution_v03, MPIData, MPIOperator};

/// Build the MPI-distributed short-range `rimatr_sr` for RSH functionals.
///
/// Returns `(rimatr_sr, basbas2baspar, baspar2basbas)` where `rimatr_sr` is of shape
/// `[n_baspar, naux_local]`: all basis-pair rows, but only the auxiliary-basis columns
/// assigned to this rank — the same layout as the full-range MPI `rimatr`.
///
/// This is a collective operation: all ranks must call it with consistent arguments.
pub fn prepare_rimatr_sr_distributed(
    mol: &Molecule,
    omega_libcint: f64,
    mpi_operator: &MPIOperator,
    mpi_data: &MPIData,
) -> (MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>) {
    // SR 2c-2e Coulomb metric (P|erfc(ωr12)/r12|Q), followed by its -1/2 power.
    // Note `Molecule::int_ij_aux_columb_with_omega` expects the (positive) functional
    // omega, i.e. the negation of the libcint convention used here.
    let aux_v = mol.int_ij_aux_columb_with_omega(-omega_libcint);
    let aux_v = _power_rayon_for_symmetric_matrix(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();

    let (basbas2baspar, baspar2basbas) = mol.prepare_baspair_map();

    let (auxbas_distribution, baspar_distribution) = match (&mpi_data.auxbas, &mpi_data.baspar) {
        (Some(auxbas), Some(baspar)) => (auxbas, baspar),
        _ => panic!(
            "The MPI distribution for rimatr is not initialized; \
             `MPIData::distribute_rimatr_tasks` must be called first."
        ),
    };

    let n_baspar = mol.num_basis * (mol.num_basis + 1) / 2;
    let my_rank = mpi_operator.rank;
    let mut ri3fn = MatrixFull::new([n_baspar, auxbas_distribution[my_rank].len()], 0.0);

    let (baspar, sbsh, ebsh) = &baspar_distribution[my_rank];
    let loc_ri3fn = if baspar.len() > 0 {
        rimatr_sr_slot(mol, &aux_v, omega_libcint, *sbsh, *ebsh)
    } else {
        MatrixFull::empty()
    };

    // exchange the blocks: each rank owns (its baspar rows × all aux) after the slot
    // computation, and receives the slices for all baspar rows × its aux columns
    mpi_isend_irecv_wrt_distribution_v03(
        &mpi_operator.world,
        &mut ri3fn,
        loc_ri3fn.data_ref().unwrap(),
        auxbas_distribution,
        baspar_distribution,
        loc_ri3fn.size()[0],
    );

    (ri3fn, basbas2baspar, baspar2basbas)
}

/// Compute this rank's local block of the solved SR `rimatr_sr`:
/// rows = basis pairs covered by AO shells `[sbsh..=ebsh]`, columns = all auxiliary
/// functions; solved against the SR `J^(-1/2)` factor `aux_v`.
///
/// This is a port of `Molecule::prepare_rimatr_for_ri_v_mpi_slot` with the
/// range-separated Coulomb kernel switched on in every (thread-local) libcint instance.
fn rimatr_sr_slot(
    mol: &Molecule,
    aux_v: &MatrixFull<f64>,
    omega_libcint: f64,
    sbsh: usize,
    ebsh: usize,
) -> MatrixFull<f64> {
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas = mol.num_auxbas;

    let basis_start = mol.cint_fdqc[sbsh][0];
    let s_baspar = (basis_start + 1) * basis_start / 2;
    let basis_end = mol.cint_fdqc[ebsh][0] + mol.cint_fdqc[ebsh][1] - 1;
    let e_baspar = (basis_end + 1) * basis_end / 2 + basis_end;
    let loc_n_baspar = e_baspar - s_baspar + 1;

    let mut tmp_ri3fn = MatrixFull::new([n_auxbas, loc_n_baspar], 0.0);
    let (sender, receiver) = channel();
    (sbsh..ebsh + 1).collect::<Vec<usize>>().par_iter().for_each_with(sender, |s, gj| {
        omp_set_num_threads_wrapper(1);

        let bas_j = *gj;
        // first, initialize rust_cint for each rayon thread, and switch to the SR kernel
        let mut cint_data = mol.initialize_cint(true);
        cint_data.set_omega(omega_libcint);

        let basis_start_j = mol.cint_fdqc[bas_j][0];
        let basis_len_j = mol.cint_fdqc[bas_j][1];
        let global_start = (basis_start_j + 1) * basis_start_j / 2;

        let mut pair_length = 0_usize;
        for bas_i in 0..bas_j + 1 {
            let basis_len_i = mol.cint_fdqc[bas_i][1];
            if bas_i != bas_j {
                pair_length += basis_len_i * basis_len_j;
            } else {
                pair_length += (basis_len_i + 1) * basis_len_i / 2;
            };
        }

        let mut loc_rimatr = MatrixFull::new([n_auxbas, pair_length], 0.0);
        for bas_i in 0..bas_j + 1 {
            let basis_start_i = mol.cint_fdqc[bas_i][0];
            let basis_len_i = mol.cint_fdqc[bas_i][1];
            mol.cint_aux_fdqc.iter().enumerate().for_each(|(k, bas_info)| {
                let basis_start_k = bas_info[0];
                let basis_len_k = bas_info[1];
                let gk = k + n_basis_shell;

                let buf = MatrixFull::from_vec(
                    [basis_len_i * basis_len_j, basis_len_k],
                    cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32),
                )
                .unwrap();

                if bas_i < bas_j {
                    for loc_j in (0..basis_len_j) {
                        let index_s = loc_j * basis_len_i;
                        let gj = loc_j + basis_start_j;
                        let loc_x_start = (gj + 1) * gj / 2 - global_start;
                        for loc_i in (0..basis_len_i) {
                            let index = index_s + loc_i;
                            let gi = loc_i + basis_start_i;
                            let loc_x = loc_x_start + gi;
                            let tmp_aux = buf.iter_row(index);
                            loc_rimatr.slice_column_mut(loc_x)
                                [basis_start_k..basis_start_k + basis_len_k]
                                .iter_mut()
                                .zip(tmp_aux)
                                .for_each(|(to, from)| *to = *from)
                        }
                    }
                } else {
                    for loc_j in (0..basis_len_j) {
                        let index_s = loc_j * basis_len_i;
                        let gj = loc_j + basis_start_j;
                        let loc_x_start = (gj + 1) * gj / 2 - global_start;
                        for loc_i in (0..loc_j + 1) {
                            let index = index_s + loc_i;
                            let gi = loc_i + basis_start_i;
                            let loc_x = loc_x_start + gi;
                            let tmp_aux = buf.iter_row(index);
                            loc_rimatr.slice_column_mut(loc_x)
                                [basis_start_k..basis_start_k + basis_len_k]
                                .iter_mut()
                                .zip(tmp_aux)
                                .for_each(|(to, from)| *to = *from)
                        }
                    }
                }
            });
        }

        cint_data.final_c2r();

        s.send((loc_rimatr, global_start - s_baspar, pair_length)).unwrap();
    });

    receiver.into_iter().for_each(|(loc_rimatr, global_start, pair_length)| {
        tmp_ri3fn
            .par_iter_columns_mut(global_start..global_start + pair_length)
            .unwrap()
            .zip(loc_rimatr.par_iter_columns_full())
            .for_each(|(to, from)| to.copy_from_slice(from))
    });

    // solve against the SR J^(-1/2): ri3fn[basis pair, aux] = Σ_P tmp[P, pair] * aux_v[P, aux]
    let mut ri3fn = MatrixFull::new([loc_n_baspar, n_auxbas], 0.0);
    omp_set_num_threads_wrapper(mol.ctrl.num_threads.unwrap());
    _dgemm_full(&tmp_ri3fn, 'T', aux_v, 'N', &mut ri3fn, 1.0, 0.0);

    ri3fn
}
