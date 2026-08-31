use std::iter::zip;
use std::ops::{Range, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign};

/// MPI construction of the short-range (RSH) decomposed 3-center RI integrals (`rimatr_sr`).
#[cfg(feature = "mpi")]
pub mod rimatr_sr;

/// Sandbox: distributed (ScaLAPACK) Cholesky factorization and triangular solve of the
/// 2c-2e metric for MPI-parallel rimatr builds (only with `mpi` + `scalapack` features).
#[cfg(all(feature = "mpi", feature = "scalapack"))]
pub mod j2c_distributed;

#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::request::WaitGuard;
#[cfg(feature = "mpi")]
use mpi::topology::{SimpleCommunicator, CartesianCommunicator, Rank, Color, Key};
#[cfg(feature = "mpi")]
use mpi::datatype::{Partitioned, PartitionMut};
#[cfg(feature = "mpi")]
use mpi::traits::*;
use num_traits::{One, Zero};
use tensors::matrix_blas_lapack::{_dgemm_full, _dsyevd, _power_rayon_for_symmetric_matrix};
use tensors::{BasicMatrix, MatrixFull};
use std::fmt::Debug;

#[cfg(feature = "scalapack")]
use tensors::matrix_scalapack::CblacsGrid;

use crate::dft::Grids;
use crate::constants::MPI_CHUNK;
use crate::molecule_io::Molecule;
use crate::utilities::balancing;

/// ==== Memory distribution for the MPI implementation in REST ====
/// There are two intermediate objects are distributed during the MPI processors
/// 1) scf_io::<SCF>.grids {
///     coordinates: vec![<[f64;3]>; num_grids], contains the coordinates of all numerical grids (num_grids).
///     weights: vec![f64; num_grids], contains the integral weights of these numerical grids (num_grids).
///     ao: MatrixFull[num_grids, num_bas], tabulates all atomic orbital basis (num_bas) according to these numerical grids (num_grids).
///     aop: RIFull[num_grids, num_bas], tabulates the derivatives of all atomic orbital basis (num_bas) according to these numerical grids (num_grids).
///   } 
///          
///   The numerical grids (num_grids) are distributed across mpi processors. Such that each processor only stores a part of grids (loc_num_grids):
///    local_grids {
///     coordinates: vec![<[f64;3]>; loc_num_grids]
///     weights: vec![f64; loc_ num_grids]
///     ao: MatrixFull[loc_num_grids, num_orbs]
///     aop: RIFull[loc_num_grids, num_orbs]
///   } 
/// 
/// 2) scf_io::<SCF>.ri3fn: MatrixFull[num_auxbas, num_baspar]
///   The auxiliary basis (num_auxbas) are distributed across mpi processors. 
///   Such that each processor only stores a part of auxiliary basis functions (loc_num_aux):
///     local_ri3fn: MatrixFull[loc_num_auxbas, num_baspar]
///   NOTE:: In the generation of local_ri3fn, num_baspar is also distributed
/// 
/// 
/// In consequence, ri3mo is also distrubted across mpi processors.
///    
/// 

#[cfg(feature = "mpi")]
pub struct MPIOperator {
    #[cfg(feature = "scalapack")]
    pub cblacsgrid: CblacsGrid,
    pub universe: Universe,
    pub world: SimpleCommunicator,
    pub size: usize,
    pub rank: usize,
}

#[cfg(not(feature = "mpi"))]
pub struct MPIOperator {
    pub size: usize,
    pub rank: usize,
}

#[cfg(feature = "mpi")]
pub struct MPIGrid {
    pub cart_comm: CartesianCommunicator,
    pub row_comm: SimpleCommunicator,
    pub col_comm: SimpleCommunicator,
    pub dims: (i32, i32),          // (r, c)
    pub my_coords: (i32, i32),     // (my_row, my_col)
    pub rank: i32,                   // rank in grid communicator
}

#[cfg(feature = "mpi")]
impl MPIOperator {

    pub fn initialize_grid(&self) -> MPIGrid {
        let size = self.size as i32;
        let (r, c) = {
            let mut r = (size as f64).sqrt() as i32;
            while size % r != 0 {
                r -= 1;
            }
            (r, size / r)
        };

        let dims = [r, c];
        let periods = [false, false];
        let cart_comm = self.world.create_cartesian_communicator(&dims, &periods, false)
                                 .expect("Failed to create Cartesian grid");

        let rank = cart_comm.rank();
        let coords = cart_comm.rank_to_coordinates(rank);

        let my_row = coords[0];
        let my_col = coords[1];

        let row_color = Color::with_value(my_row);
        let row_key: Key = my_col;
        let col_color = Color::with_value(my_col);
        let col_key: Key = my_row;

        let row_comm = cart_comm
                .split_by_color_with_key(row_color, row_key)
                .expect("Failed to split row communicator");

        let col_comm = cart_comm
               .split_by_color_with_key(col_color, col_key)
               .expect("Failed to split column communicator");

        MPIGrid {
            cart_comm,
            row_comm,
            col_comm,
            dims: (r, c),
            my_coords: (my_row, my_col),
            rank,
        }

    }

}

#[derive(Clone)]
pub struct MPIData {
    pub size: usize,
    pub rank: usize,
    pub grids: Option<Vec<Range<usize>>>,
    pub auxbas: Option<Vec<Range<usize>>>,
    pub baspar: Option<Vec<(Range<usize>, usize, usize)>>
}

impl MPIData {
    #[cfg(feature = "mpi")]
    pub fn initialization() -> (Option<MPIOperator>,Option<MPIData>) {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size() as usize;
        let rank = world.rank() as usize;
        #[cfg(feature = "scalapack")]
        let cgrid = CblacsGrid::new(size, "R");

        if size >= 2 {
            (
                Some(MPIOperator {
                universe,
                world,
                size,
                rank,
                #[cfg(feature = "scalapack")]
                cblacsgrid: cgrid,
                }),
                Some(MPIData{
                    size,
                    rank,
                    grids: None,
                    auxbas: None,
                    baspar: None
                })
            )
        } else {
            (None, None)
        }
    }

    #[cfg(not(feature = "mpi"))]
    pub fn initialization() -> (Option<MPIOperator>,Option<MPIData>) {
        (None, None)
    }

    pub fn distribute_grids_tasks(&mut self, grids: &Grids) -> Grids {
        let num_tasks = grids.weights.len();

        let mut distribute_vec = average_distribution(num_tasks, self.size) ;
        
        let local_range = &distribute_vec[self.rank];
        let local_coordinates = grids.coordinates[local_range.clone()].to_vec();
        let local_weights = grids.weights[local_range.clone()].to_vec();
        let local_atm_idx = grids.atm_idx[local_range.clone()].to_vec();
        let local_quadrature_weights = grids.quadrature_weights[local_range.clone()].to_vec();
        self.grids = Some(distribute_vec);

        let parallel_balancing = balancing(local_coordinates.len(), rayon::current_num_threads());
        return Grids {
            coordinates: local_coordinates,
            weights: local_weights,
            ao: None,
            aop: None,
            parallel_balancing,
            non0tab: None,
            ao_cutoff: grids.ao_cutoff,
            ao_compressed: None,
            aop_compressed: None,
            atm_idx: local_atm_idx,
            quadrature_weights: local_quadrature_weights,
        }
        
    }

    pub fn distribution_elec_pair(&self, start_mo: usize, num_occu: usize) -> Vec<[usize;2]> {

        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in start_mo..num_occu {
            for j_state in i_state..num_occu {
                elec_pair.push([i_state,j_state])
            }
        };

        let distribution_vec = average_distribution(elec_pair.len(), self.size);

        let loc_elec_pair = elec_pair[distribution_vec[self.rank].clone()].to_vec();

        loc_elec_pair

    }

    pub fn distribution_same_spin_virtual_orbital_pair(&self, lumo: usize, num_state: usize) -> Vec<[usize;2]> {

        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in lumo..num_state {
            for j_state in i_state+1..num_state {
                elec_pair.push([i_state,j_state])
            }
        };

        let distribution_vec = average_distribution(elec_pair.len(), self.size);

        let loc_elec_pair = elec_pair[distribution_vec[self.rank].clone()].to_vec();

        loc_elec_pair

    }

    pub fn distribution_opposite_spin_virtual_orbital_pair(&self, lumo_1: usize, lumo_2: usize, num_state: usize, parallel_mode: usize) -> Vec<[usize;2]> {

        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in lumo_1..num_state {
            for j_state in lumo_2..num_state {
                elec_pair.push([i_state,j_state])
            }
        };

        if parallel_mode == 0 {
            let distribution_vec = average_distribution(elec_pair.len(), self.size);
            let loc_elec_pair = elec_pair[distribution_vec[self.rank].clone()].to_vec();
            loc_elec_pair
        } else {
            if self.rank == 0 {
                elec_pair
            } else {
                vec![]
            }
        }
    }

    pub fn distribute_rimatr_tasks(&mut self, num_auxbas: usize, num_basis: usize, cint_bas: Vec<Vec<usize>>,) {

        self.auxbas = Some(average_distribution(num_auxbas, self.size));


        // now distribute the basis pairs
        let num_basis = num_basis;
        let num_baspar = (num_basis+1)*num_basis/2;
        let n_basis_shell = cint_bas.len();

        let chunk_size = num_baspar/self.size;
        let chunk_rest = num_baspar%self.size;

        //println!("debug: num_basis: {}, num_baspar: {}, n_basis_shell: {}", num_basis, num_baspar, n_basis_shell);

        let (basbas2baspar , baspar2basbas) = prepare_baspair_map(num_basis);
        let mut distribute_vec = vec![(0..num_baspar, 0_usize, n_basis_shell);self.size];
        let mut start = 0_usize;
        let mut count = chunk_rest  as i32;
        distribute_vec.iter_mut().enumerate().for_each(|(i,(data, sbsh, ebsh))| {
            if count >0 {
                *data = start..start + chunk_size+1;
                start += chunk_size+1;
                count -= 1;
            } else {
                *data = start..start + chunk_size;
                start += chunk_size;
                //count -= 1;
            }
        });

        // reshape the basis pairs distribution according to basis_shells
        distribute_vec.iter_mut().for_each(|(data, sbsh, ebsh)| {
            let s_basis = baspar2basbas[data.start];
            let e_basis = baspar2basbas[data.end-1];
            let last_bsh = cint_bas.last().unwrap();


            let (s_bsh, e_bsh) = cint_bas.iter().enumerate()
            .fold((0, 0), |acc, (i_bsh,bsh)| {

                let (mut s_bsh, mut e_bsh) = acc;

                if s_basis[1] > bsh[0] && s_basis[1] < bsh[0]+bsh[1] {
                    s_bsh = (i_bsh+1) as usize;
                } else if s_basis[1] == bsh[0] {
                    s_bsh = i_bsh as usize;
                }
                if e_basis[1] >= bsh[0] && e_basis[1] < bsh[0]+bsh[1] {
                    e_bsh = i_bsh as usize;
                }
                (s_bsh, e_bsh)
            });

            //let s_bas = &cint_bas[s_bsh];
            //let e_bas = &cint_bas[e_bsh];
            *sbsh = s_bsh;
            *ebsh = e_bsh;

            if s_bsh < n_basis_shell {

                let s_bas = &cint_bas[s_bsh];
                let e_bas = &cint_bas[e_bsh];

                let s_baspar = basbas2baspar[[0, s_bas[0] as usize]];//((s_bas[0]+1)*s_bas[0]/2) as usize; //
                let e_baspar = basbas2baspar[[(e_bas[0] + e_bas[1]-1) as usize, (e_bas[0] + e_bas[1]-1) as usize]];

                *data = s_baspar .. e_baspar+1;

            } else {
                *sbsh = n_basis_shell-1;
                *data = num_baspar .. num_baspar;
            }

            //println!("debug after: baspar_range {:?} s_basis {:?} e_basis {:?} s_bsh {} e_bsh {}", &data, &s_basis, &e_basis, s_bsh, e_bsh);

        });

        self.baspar = Some(distribute_vec)



    }

}

pub fn prepare_baspair_map(n_basis: usize) -> (MatrixFull<usize>, Vec<[usize;2]>) {
    let n_baspar = (n_basis+1)*n_basis/2;

    let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
    let mut baspar2basbas = vec![[0_usize;2];n_baspar];
    basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
        &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
            let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
            *map_v = baspar_ind;
            baspar2basbas[baspar_ind] = [i,j];
        });
    });

    (basbas2baspar, baspar2basbas)
}

#[cfg(feature = "mpi")]
pub fn mpi_reduce<Q>(world: &SimpleCommunicator, data: &[Q], root_rank: usize, op: &SystemOperation) -> Vec<Q> 
where Q: Add<Output=Q> + AddAssign + 
         Sub<Output=Q> + SubAssign + 
         Mul<Output=Q> + MulAssign + 
         Div<Output=Q> + DivAssign + 
         Zero + Send + Sync + Copy + Debug + Buffer + 'static,
      [Q]: Buffer,
      Vec<Q>: BufferMut
      
{
    let rank = world.rank() as usize;
    let root_process = world.process_at_rank(root_rank as i32);

    //let mut result: Vec<Q> = if rank == root_rank {vec![Q::zero(); data.len()]} else {vec![]};
    let mut result: Vec<Q> = vec![Q::zero(); data.len()];

    //println!("debug rank {} and data {:?} in mpi_reduce before", rank, &data);

    if rank == root_rank {
        root_process.reduce_into_root(&data[..], &mut result, op);
    } else {
        root_process.reduce_into(&data[..], op);
    }

    //world.barrier();

    result
}

#[cfg(feature = "mpi")]
pub fn mpi_allreduce<Q>(world: &SimpleCommunicator, data: &[Q], reduced_data: &mut [Q], op: &SystemOperation)
where Q: Add<Output=Q> + AddAssign + 
         Sub<Output=Q> + SubAssign + 
         Mul<Output=Q> + MulAssign + 
         Div<Output=Q> + DivAssign + 
         Zero + Send + Sync + Copy + Debug + Buffer + 'static,
      [Q]: Buffer + BufferMut,
      Vec<Q>: BufferMut
      
{

    let batch = communicate_by_batch(data.len());

    for i_batch in batch {
        world.any_process().all_reduce_into(&data[i_batch.clone()], &mut reduced_data[i_batch.clone()], op);
    }

    //world.barrier();
}


#[cfg(feature = "mpi")]
pub fn mpi_broadcast<Q>(world: &SimpleCommunicator, data: &mut Q, root_rank: usize)
where Q: Send + Sync + Buffer + Debug + BufferMut + 'static,
{

    let root_process = world.process_at_rank(root_rank as i32);
    root_process.broadcast_into(data);

    //let rank = world.rank();
    //println!("debug Rank {} received value: {:?}", rank, &data);

}

#[cfg(feature = "mpi")]
pub fn mpi_broadcast_vector<Q>(world: &SimpleCommunicator, data: &mut Vec<Q>, root_rank: usize)
where Q: Zero + Send + Sync + Copy + Buffer + Equivalence + Debug + 'static,
      Vec<Q>: BufferMut
{
    world.barrier();
    let rank = world.rank() as usize;
    //println!("debug rank {}", rank);
    let mut data_len = data.len();
    mpi_broadcast::<usize>(world, &mut data_len, root_rank);
    if data_len != 0 {
        //println!("debug data_len: {} in rank {}", data_len, rank);
        if rank!= root_rank {
            data.resize(data_len, Q::zero());
        }
        mpi_broadcast::<Vec<Q>>(world, data, root_rank);
    }
}

#[cfg(feature = "mpi")]
pub fn mpi_broadcast_matrixfull<Q>(world: &SimpleCommunicator, data: &mut MatrixFull<Q>, root_rank: usize)
where Q: Zero + Send + Sync + Copy + Buffer + Equivalence + Debug + 'static,
      [Q]: BufferMut
{
    let rank = world.rank() as usize;
    let mut data_size = data.size.clone();
    mpi_broadcast::<[usize;2]>(world, &mut data_size, root_rank);
    if data_size[0]*data_size[1] != 0 {
        if rank!= root_rank {
            *data = MatrixFull::new(data_size, Q::zero());
        }
        mpi_broadcast(world, &mut data.data, root_rank);
    }
}

/// All-gather a matrix whose column dimension is distributed across MPI ranks.
///
/// Each rank holds the contiguous column block `[n_row, loc_naux]` of the global
/// `[n_row, naux_total]` matrix, where its local columns are the block assigned by
/// [`average_distribution`]. Since `MatrixFull` is stored column-major, the raw data of
/// each rank is exactly the contiguous global column block, so a variable-count
/// all-gather of the local raw data (concatenated in rank order) reproduces the full
/// global matrix on every rank.
#[cfg(feature = "mpi")]
pub fn mpi_allgather_matrixfull_columns(
    world: &SimpleCommunicator,
    local: &MatrixFull<f64>,
    naux_total: usize,
) -> MatrixFull<f64> {
    let size = world.size() as usize;
    let n_row = local.size[0];
    let loc_naux = local.size[1];

    // exchange the number of columns held by each rank
    let mut loc_naux_vec = vec![0_usize; size];
    world.all_gather_into(&loc_naux, &mut loc_naux_vec[..]);

    // the distribution must be the contiguous ascending one from `average_distribution`,
    // so that concatenating the rank-ordered column blocks reproduces the global column order
    let expected = average_distribution(naux_total, size);
    assert!(
        loc_naux_vec.iter().zip(expected.iter()).all(|(n, r)| *n == r.len()),
        "The column distribution of the matrix does not match average_distribution; \
         cannot safely all-gather the full matrix."
    );

    // counts/displacements in units of f64 elements (one column holds n_row elements)
    let counts: Vec<i32> = loc_naux_vec
        .iter()
        .map(|&n| {
            i32::try_from(n * n_row)
                .expect("The column block of the matrix exceeds the MPI count limit (i32).")
        })
        .collect();
    let mut displs: Vec<i32> = Vec::with_capacity(size);
    let mut acc: i32 = 0;
    for &count in counts.iter() {
        displs.push(acc);
        acc += count;
    }

    let mut gathered = vec![0.0_f64; n_row * naux_total];
    {
        let mut partition = PartitionMut::new(&mut gathered[..], &counts[..], &displs[..]);
        world.all_gather_varcount_into(&local.data[..], &mut partition);
    }
    MatrixFull::from_vec([n_row, naux_total], gathered).unwrap()
}

/// Reconstruct the complete auxiliary-basis dimension of `rimatr` (the decomposed 3c ERI,
/// i.e. `cderi`) on every MPI rank, for consumers that require the full matrix.
///
/// In MPI-parallel SCF, `rimatr` is distributed along the auxiliary-basis (column)
/// dimension: rank `r` stores only `[n_baspar, auxbas_distribution[r].len()]` (see
/// [`MPIData::distribute_rimatr_tasks`] and
/// `Molecule::prepare_rimatr_for_ri_v_mpi_rayon`); the SCF J/K build then reduces the
/// partial contractions over ranks (`vj/vk_upper_with_rimatr_sync_mpi`). The analytical
/// gradient, in contrast, needs the complete `[n_baspar, naux]` matrix on every rank.
///
/// - Returns `Some(full_matrix)` if the input was distributed and has been gathered.
/// - Returns `None` if the input already contains all auxiliary functions (serial runs,
///   or MPI runs where the distribution is inactive), leaving the input untouched.
///
/// This is a collective operation: all ranks must call it with consistent arguments.
pub fn gather_full_rimatr(
    local_rimatr: &MatrixFull<f64>,
    naux_total: usize,
    mpi_operator: &Option<MPIOperator>,
) -> Option<MatrixFull<f64>> {
    #[cfg(feature = "mpi")]
    if let Some(mpi_op) = mpi_operator {
        if local_rimatr.size[1] != naux_total {
            return Some(mpi_allgather_matrixfull_columns(&mpi_op.world, local_rimatr, naux_total));
        }
    }
    None
}

/// Build the factor of the 2c-2e Coulomb metric used by the MPI-parallel rimatr
/// construction, following the `j2c_decomp` policy so that the result reproduces the
/// serial one (`ri_jk::generate_rimatr_bare` + `ri_jk::get_solved_j3c`):
///
/// - `Eig`: returns the dense `J^{-1/2}` (eigen-based matrix power); the serial
///   counterpart computes `cderi = j3c · J^{-1/2}` with a matrix multiplication.
/// - `Cd`:   returns the upper Cholesky factor `U` (`J = U^T U`, `L = U^T`); the serial
///   counterpart computes `cderi = j3c · L^{-T}`, which the solver reproduces with the
///   triangular solve `U^T · X = B` (`_dtrtrs`) instead of an explicit inverse.
///   When the Cholesky factor has diagonal elements not larger than the threshold (or
///   the factorization is not positive-definite), the same eigen-corrected fallback as
///   `ri_jk::decomp_j2c_cd` is applied.
pub fn prepare_j2c_solve_factor(
    j2c: &MatrixFull<f64>,
    option: &crate::ri_jk::J2CDecompOption,
) -> MatrixFull<f64> {
    use crate::ri_jk::{J2CDecompPolicy, J2C_THRESH};

    let threshold = option.threshold.unwrap_or(J2C_THRESH);
    match option.policy {
        J2CDecompPolicy::Eig => {
            _power_rayon_for_symmetric_matrix(j2c, -0.5, threshold).unwrap()
        },
        J2CDecompPolicy::Cd => {
            let n = j2c.size[0];
            debug_assert_eq!(j2c.size[1], n);
            // direct Cholesky attempt on a clone (the helpers panic on failure, so
            // guard with the eigen decomposition check below when needed)
            let mut j2c_u = j2c.clone();
            j2c_u.to_matrixfullslicemut().lapack_dpotrf(b'U');
            // dpotrf only touches the upper triangle; zero out the lower part so
            // that the factor can be used as a full matrix in the subsequent solve
            for j in 0..n {
                for i in j + 1..n {
                    j2c_u.data[i + j * n] = 0.0;
                }
            }
            let diag_ok = j2c_u
                .iter_diagonal()
                .unwrap()
                .fold(true, |acc, &d| acc && d > threshold);
            if diag_ok {
                // return the upper Cholesky factor U (J = U^T U); the solve step
                // (`U^T · X = B` via dtrtrs) replaces the explicit `U^{-1}` + dgemm
                return j2c_u;
            }
            // eigen-corrected fallback (mirrors `decomp_j2c_cd`): make the matrix
            // sufficiently positive-definite with the threshold, then Cholesky again
            let (eigvec, eigval, _) = _dsyevd(j2c, 'V');
            let eigvec = eigvec.expect("Failed to diagonalize the 2c-2e Coulomb matrix for the Cholesky fallback");
            // j2c_corr = V · diag(max(e, threshold)) · V^T
            let mut e_corr = MatrixFull::new([n, n], 0.0_f64);
            eigval.iter().enumerate().for_each(|(i, &e)| {
                *&mut e_corr.data[i * n + i] = e.max(threshold);
            });
            let mut j2c_corr = MatrixFull::new([n, n], 0.0_f64);
            _dgemm_full(&eigvec, 'N', &e_corr, 'N', &mut j2c_corr, 1.0, 0.0);
            let mut tmp = MatrixFull::new([n, n], 0.0_f64);
            _dgemm_full(&j2c_corr, 'N', &eigvec, 'T', &mut tmp, 1.0, 0.0);
            tmp.to_matrixfullslicemut().lapack_dpotrf(b'U');
            for j in 0..n {
                for i in j + 1..n {
                    tmp.data[i + j * n] = 0.0;
                }
            }
            tmp
        },
    }
}

#[cfg(feature = "mpi")]
pub fn mpi_isend_irecv_wrt_distribution<Q>(
    world: &SimpleCommunicator,
    data: &[Q],
    distribution: &[Range<usize>], 
    scale: usize) -> Vec<Vec<Q>>
where
    Q: Zero + Send + Sync + Buffer + Debug + Clone + BufferMut + 'static,
    [Q]: Buffer + BufferMut,
    Vec<Q>: BufferMut,
{
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let mut scale_vec = vec![scale; size];
    world.all_gather_into(&scale, &mut scale_vec[..]);

    let mut result: Vec<Vec<Q>> = vec![vec![]; distribution.len()];
    //let chunk_len = distribution[rank].len()*scale;
    result.iter_mut().enumerate().for_each(|(i,x)| *x = vec![Q::zero(); distribution[rank].len()*scale_vec[i]]);

    //println!("debug mpi 0 in rank {} with scale_vec = {:?}, distribution = {:?}", rank, &scale_vec, &distribution[rank]);

    //let mut rreq_vec = vec![];
    //let mut sreq_vec = vec![];
    for i in 0..size {
        for j in 0..size {
            if rank == j || rank == i {
                let tmp_range = distribution[i].start*scale_vec[j] .. distribution[i].end*scale_vec[j];
                let by_batch = communicate_by_batch(tmp_range.len());
                println!("debug ij: {} {}, by_batch_len: {}", i, j, &by_batch.len());
                by_batch.iter().enumerate().for_each(|(k,loc_batch)| {
                    mpi::request::scope(|scope| {
                        let sreq = if rank == j {
                            let tmp_range_batch = (tmp_range.start + loc_batch.start)..(tmp_range.start + loc_batch.start + loc_batch.len());
                            Some(
                                world.process_at_rank(i as i32).immediate_send(scope, &data[tmp_range_batch])
                            )
                        } else {None};
                        if rank == i {
                            let result_slice = &mut result[j][by_batch[k].clone()];//.as_mut_slice();
                            let rreq = world.process_at_rank(j as i32).immediate_receive_into(scope, result_slice);
                            rreq.wait();
                        } 
                        if rank == j {
                            if let Some(sreq_j) = sreq {
                                sreq_j.wait();
                            }
                        }
                    });
                });
            }

            //mpi::request::scope(|scope| {
            //    let sreq = if rank == j {
            //        let tmp_range = distribution[i].start*scale_vec[j] .. distribution[i].end*scale_vec[j];
            //        Some(
            //            world.process_at_rank(i as i32).immediate_send(scope, &data[tmp_range])
            //        )
            //    } else {
            //        None
            //    };
            //    if rank == i {
            //        let result_slice = result[j].as_mut_slice();
            //        let rreq = world.process_at_rank(j as i32).immediate_receive_into(scope, result_slice);
            //        rreq.wait();
            //    } 
            //    if rank == j {
            //        if let Some(sreq_j) = sreq {
            //            sreq_j.wait();
            //        }
            //    }
            //});
        }
    }

    //for rreq_i in rreq_vec {
    //    rreq_i.wait();
    //}

    //for sreq_i in sreq_vec {
    //    if let Some(sreq_i) = sreq_i {
    //        sreq_i.wait();
    //    }
    //}

    result
}


pub fn average_distribution(num_tasks: usize, size: usize) -> Vec<Range<usize>> {
    let mut distribute_vec: Vec<Range<usize>> = vec![0..num_tasks;size];
    
    let chunk_size = num_tasks/size;
    let chunk_rest = num_tasks%size;

    let mut start = 0_usize;
    let mut count = chunk_rest  as i32;
    distribute_vec.iter_mut().enumerate().for_each(|(i,data)| {
        if count >0 {
            *data = start..start + chunk_size+1;
            start += chunk_size+1;
            count -= 1;
        } else {
            *data = start..start + chunk_size;
            start += chunk_size;
            //count -= 1;
        }
    });

    distribute_vec
}

pub fn average_distribution_with_residue(num_tasks: usize, size: usize) -> (Vec<Range<usize>>, Range<usize>) {
    let mut distribute_vec: Vec<Range<usize>> = vec![0..num_tasks;size];
    
    let chunk_size = num_tasks/size;
    let chunk_rest = num_tasks%size;

    let mut start = 0_usize;
    distribute_vec.iter_mut().enumerate().for_each(|(i,data)| {
            *data = start..start + chunk_size;
            start += chunk_size;
    });

    (distribute_vec, start..num_tasks)
}

pub fn communicate_by_batch(num_tasks: usize) -> Vec<Range<usize>> {
    let mut chunk_size = MPI_CHUNK; // around 1 Gb
    //let mut chunk_size = 16_usize; 
    
    let chunk_rest = num_tasks%chunk_size;
    let num_comm = if chunk_rest==0 {num_tasks/chunk_size} else {num_tasks/chunk_size+1};

    let mut distribute_vec: Vec<Range<usize>> = vec![0..num_tasks;num_comm];

    let mut start = 0_usize;
    distribute_vec.iter_mut().enumerate().for_each(|(i,data)| {
        let num_length = if start + chunk_size > num_tasks {
            num_tasks - start
        } else {
            chunk_size
        };

        *data = start..start + num_length;
        start += num_length;
    });

    distribute_vec
}

#[cfg(feature = "mpi")]
pub fn mpi_isend_irecv_wrt_distribution_v02<Q>(
    world: &SimpleCommunicator,
    data: &[Q],
    distribution: &[Range<usize>], 
    scale: usize) -> Vec<Vec<Q>>
where
    Q: Zero + Send + Sync + Buffer + Debug + Clone + BufferMut + 'static,
    [Q]: Buffer + BufferMut,
    Vec<Q>: BufferMut,
{
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let mut scale_vec = vec![scale; size];
    world.all_gather_into(&scale, &mut scale_vec[..]);

    let mut result: Vec<Vec<Q>> = vec![vec![]; distribution.len()];
    //let chunk_len = distribution[rank].len()*scale;
    result.iter_mut().enumerate().for_each(|(i,x)| *x = vec![Q::zero(); distribution[rank].len()*scale_vec[i]]);


    //println!("debug mpi 0 in rank {} with scale_vec = {:?}, distribution = {:?}", rank, &scale_vec, &distribution[rank]);

    //let mut rreq_vec = vec![];
    //let mut sreq_vec = vec![];
    for i in 0..size {
        for j in 0..size {
            if rank == j || rank == i {
                let tmp_range = distribution[i].start*scale_vec[j] .. distribution[i].end*scale_vec[j];
                let by_batch = communicate_by_batch(tmp_range.len());
                let count = by_batch.len();
                let mut result_j  = Vec::new();
                let mut result_j_left = result[j].as_mut_slice();
                let mut local_start = 0;
                if rank == i {
                    for (k, range_k) in by_batch.iter().enumerate() {
                        let (result_j_chunk, result_j_left_2) = result_j_left.split_at_mut(range_k.end-local_start);
                        result_j.push(result_j_chunk);
                        local_start = range_k.end;
                        result_j_left = result_j_left_2;
                    };
                }
                //println!("debug ij: {} {}, by_batch_len: {}", i, j, &by_batch.len());
                mpi::request::multiple_scope(count*2, |scope, coll| {

                    if rank == j {
                        for ((k, loc_batch)) in by_batch.iter().enumerate() {
                            let tmp_range_batch = (tmp_range.start + loc_batch.start)..(tmp_range.start + loc_batch.start + loc_batch.len());
                            let sreq = world
                                .process_at_rank(i as i32)
                                .immediate_send(scope, &data[tmp_range_batch]);
                            coll.add(sreq);
                        }
                    }
                    if rank == i {
                        for ((k, loc_batch), result_jk) in zip(by_batch.iter().enumerate(), result_j) {
                            let rreq = world
                                .process_at_rank(j as i32)
                                .immediate_receive_into(scope, result_jk);
                            coll.add(rreq);
                        }
                    };
                    while coll.incomplete() > 0 {
                        coll.test_any();
                    }
                });
            }
        }
    }

    result
}

#[cfg(feature = "mpi")]
pub fn mpi_isend_irecv_wrt_distribution_v03<Q>(
    world: &SimpleCommunicator,
    outdata: &mut MatrixFull<Q>,
    data: &[Q],
    auxbas_distribution: &[Range<usize>],
    baspar_distribution: &[(Range<usize>,usize, usize)], 
    scale: usize)
where
    Q: Zero + Clone + Copy + Send + Sync + Buffer + Debug + Clone + BufferMut + 'static,
    [Q]: Buffer + BufferMut,
    Vec<Q>: BufferMut,
{
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let mut scale_vec = vec![scale; size];
    world.all_gather_into(&scale, &mut scale_vec[..]);

    //let mut result: Vec<Vec<Q>> = vec![vec![]; distribution.len()];
    //let chunk_len = distribution[rank].len()*scale;
    //result.iter_mut().enumerate().for_each(|(i,x)| *x = vec![Q::zero(); distribution[rank].len()*scale_vec[i]]);


    //println!("debug mpi 0 in rank {} with scale_vec = {:?}, distribution = {:?}", rank, &scale_vec, &distribution[rank]);

    //let mut rreq_vec = vec![];
    //let mut sreq_vec = vec![];
    for i in 0..size {
        for j in 0..size {
            if rank == j || rank == i {
                let mut result = if rank == i {vec![Q::zero(); auxbas_distribution[i].len()*scale_vec[j]]} else {Vec::new()};

                let tmp_range = auxbas_distribution[i].start*scale_vec[j] .. auxbas_distribution[i].end*scale_vec[j];
                let by_batch = communicate_by_batch(tmp_range.len());
                let count = by_batch.len();

                let mut result_j  = Vec::new();
                let mut result_j_left = result.as_mut_slice();
                let mut local_start = 0;
                if rank == i {
                    for (k, range_k) in by_batch.iter().enumerate() {
                        let (result_j_chunk, result_j_left_2) = result_j_left.split_at_mut(range_k.end-local_start);
                        result_j.push(result_j_chunk);
                        local_start = range_k.end;
                        result_j_left = result_j_left_2;
                    };
                }
                //println!("debug ij: {} {}, by_batch_len: {}", i, j, &by_batch.len());
                mpi::request::multiple_scope(count*2, |scope, coll| {
                    if rank == j {
                        for ((k, loc_batch)) in by_batch.iter().enumerate() {
                            let tmp_range_batch = (tmp_range.start + loc_batch.start)..(tmp_range.start + loc_batch.start + loc_batch.len());
                            let sreq = world
                                .process_at_rank(i as i32)
                                .immediate_send(scope, &data[tmp_range_batch]);
                            coll.add(sreq);
                        }
                    }
                    if rank == i {
                        for ((k, loc_batch), result_jk) in zip(by_batch.iter().enumerate(), result_j) {
                            let rreq = world
                                .process_at_rank(j as i32)
                                .immediate_receive_into(scope, result_jk);
                            coll.add(rreq);
                        }
                    };
                    while coll.incomplete() > 0 {
                        coll.test_any();
                    }
                });

                if rank == i  {
                    let (baspar, sbsh, ebsh) = &baspar_distribution[j];
                    outdata.iter_submatrix_mut(baspar.clone(), 0..auxbas_distribution[i].len())
                    .zip(result.iter()).for_each(|(to, from)| {*to = *from});
                }
            }
        }
    }

    //result
}