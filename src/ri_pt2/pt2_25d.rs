use crate::mpi_io::*;
use rest_tensors::ri::*;
use mpi::collective::SystemOperation;

use mpi::traits::*;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::{SimpleCommunicator};

use std::mem;
use std::env;
use std::ops::Range;
use rest_tensors::RIFull;
use rest_tensors::{TensorOpt, MatrixFull, MatrixFullSlice, BasicMatrix};
use rest_tensors::matrix_blas_lapack::{_dgemm};
use libc::{sysconf, _SC_PHYS_PAGES};

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

pub struct Ctx25dBlock {
    pub k: usize,
    pub block_start_idx: Vec<usize>,
    pub ownership: Vec<Vec<usize>>,
    pub row_block_idx: Vec<usize>,
    pub col_block_idx: Vec<usize>,
    pub local_block_idx: Vec<usize>,
    pub block_idx_sent_row: Vec<Vec<usize>>,
    pub block_idx_sent_col: Vec<Vec<usize>>,
}

pub fn initialize_metadata(grid: &MPIGrid, n2_global: usize) -> Ctx25dBlock {
    let (r, c) = grid.dims;
    let r = r as usize;
    let c = c as usize;
    let my_row = grid.my_coords.0 as usize;
    let my_col = grid.my_coords.1 as usize;

    let grid_mult = if r == c {r} else {r * c};
    let target_block_size: usize = 8;
    let min_blocks = (n2_global + target_block_size - 1) / target_block_size;
    let k = std::cmp::max(2, (min_blocks + grid_mult - 1) / grid_mult);
    let num_block = k * grid_mult;
    let block_size = n2_global / num_block;
    let block_residue = n2_global % num_block;
    assert!(block_size >= 1, "N2 too small for 2.5d method");

    let mut block_start_idx: Vec<usize> = vec![0; num_block];
    for i in 1..num_block {
        let cur_block_size = if i <= block_residue {block_size + 1} else {block_size};
        block_start_idx[i] = block_start_idx[i - 1] + cur_block_size;
    }

    let ownership = determine_block_ownership(grid, num_block);

    let row_comm_size = grid.row_comm.size() as usize;
    let col_comm_size = grid.col_comm.size() as usize;
    let mut row_block_idx = Vec::new();
    let mut col_block_idx = Vec::new();
    let mut local_block_idx = Vec::new();
    let mut block_idx_sent_row = vec![Vec::new(); row_comm_size];
    let mut block_idx_sent_col = vec![Vec::new(); col_comm_size];

    for blk in 0..num_block {
        let i = blk % r;
        let j = blk % c;
        if i == my_row {
            row_block_idx.push(blk);
            block_idx_sent_row[j].push(blk);
            local_block_idx.push(blk);
        }
        if j == my_col {
            col_block_idx.push(blk);
            block_idx_sent_col[i].push(blk);
            if i != my_row {
                local_block_idx.push(blk);
            }
        }
    }

    Ctx25dBlock {
        k,
        block_start_idx,
        ownership,
        row_block_idx,
        col_block_idx,
        local_block_idx,
        block_idx_sent_row,
        block_idx_sent_col,
    }
}

fn determine_block_ownership(grid: &MPIGrid, num_block: usize) -> Vec<Vec<usize>> {
    let (r, c) = grid.dims;
    let r = r as usize;
    let c = c as usize;
    let size = grid.cart_comm.size() as usize;
    let mut ownership: Vec<Vec<usize>> = vec![Vec::new(); size];

    for blk in 0..num_block {
        let i = blk % r;
        let j = blk % c;
        let rank = i * c + j;
        ownership[rank].push(blk);
    }
    ownership
}

pub fn check_memory_25d(grid: &MPIGrid, ctx: &Ctx25dBlock, slice_size: usize, n2_global: usize) -> bool {
    let block_start_idx = &ctx.block_start_idx;
    let local_block_idx = &ctx.local_block_idx;
    let num_block = block_start_idx.len();
    let mut local_num_slice: usize = 0;
    for &blk in local_block_idx {
        let num_slice_cur_block = if blk == num_block - 1 {n2_global - block_start_idx[blk]} else {block_start_idx[blk + 1] - block_start_idx[blk]};
        local_num_slice += num_slice_cur_block;
    }
    let elem_size = std::mem::size_of::<f64>();
    let required_bytes = local_num_slice * slice_size * elem_size;
    let avail_bytes = get_available_memory_bytes();
    let required_mib = required_bytes as f64 / (1024.0 * 1024.0);
    let avail_mib = avail_bytes as f64 / (1024.0 * 1024.0);
    let local_mem_flag = required_mib < avail_mib;
    let mut global_mem_flag: bool = true;
    grid.cart_comm.all_reduce_into(&local_mem_flag, &mut global_mem_flag, &SystemOperation::logical_and());
    return global_mem_flag;
}

fn get_available_memory_bytes() -> u64 {
    // _SC_AVPHYS_PAGES: number of available physical pages
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
    let available_pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) as u64 };
    if available_pages == u64::MAX {
        panic!("Failed to get available memory");
    }
    page_size * available_pages
}


fn redistribute_to_diag(grid: &MPIGrid, ctx: &Ctx25dBlock, dten: &RIFull<f64>, n0_global: usize, n0_range: &Range<usize>) -> RIFull<f64> {
    assert!(dten.dist_axis == 0);
    let P = grid.cart_comm.size() as usize;
    let my_rank = grid.rank as usize;
    let block_start_idx = &ctx.block_start_idx;
    let ownership = &ctx.ownership;
    let num_block = block_start_idx.len() as usize;
    let n1_global = dten.size[1];
    let n2_global = dten.size[2];

    let start_n0 = n0_range.start;
    let local_n0 = n0_range.end - n0_range.start;
    let mut all_start_n0: Vec<usize> = vec![0; P];
    let mut all_local_n0: Vec<usize> = vec![0; P];
    grid.cart_comm.all_gather_into(&start_n0, &mut all_start_n0);
    grid.cart_comm.all_gather_into(&local_n0, &mut all_local_n0);

    let mut total_idx_recv = 0 as usize;
    let my_owned_blocks: &Vec<usize> = &ownership[my_rank];
    for &blk in my_owned_blocks {
        total_idx_recv += if blk == num_block - 1 {n2_global - block_start_idx[blk]} else {block_start_idx[blk + 1] - block_start_idx[blk]};
    }

    let mut send_count: Vec<usize> = vec![0; P];
    let slice_size = local_n0 * n1_global;
    for desc in 0..P {
        let mut cnt: usize = 0;
        let cur_owned_blocks: &Vec<usize> = &ownership[desc];
        for &blk in cur_owned_blocks {
            cnt += if blk == num_block - 1 {slice_size * (n2_global - block_start_idx[blk])} else {slice_size * (block_start_idx[blk + 1] - block_start_idx[blk])};
        }
        send_count[desc] = cnt;
    }

    let mut send_displs: Vec<usize> = vec![0; P];
    for i in 1..P {
        send_displs[i] = send_displs[i - 1] + send_count[i - 1];
    }
    let mut dest_ptr = send_displs.clone();
    let send_total = send_displs.last().unwrap() + send_count.last().unwrap();
    let mut send_buffer: Vec<f64> = vec![0.0f64; send_total];

    for cur_block in 0..block_start_idx.len() {
        let mut owner_tmp = -1isize;
        for i in 0..ownership.len() {
            let cur_owned_blocks: &Vec<usize> = &ownership[i];
            if cur_owned_blocks.contains(&cur_block) {
                owner_tmp = i as isize;
                break;
            }
        }
        assert!(owner_tmp != -1);
        let owner = owner_tmp as usize;
        let cur_block_idx_start = block_start_idx[cur_block];
        let cur_block_idx_end = if cur_block == num_block - 1 {n2_global - 1} else {block_start_idx[cur_block + 1] - 1};
        for cur_n2_idx in cur_block_idx_start..=cur_block_idx_end {
            let src_start = cur_n2_idx * slice_size;
            let src_slice = &dten.data[src_start..src_start + slice_size];
            let dest_start = dest_ptr[owner];
            send_buffer[dest_start..dest_start + slice_size].copy_from_slice(src_slice);
            dest_ptr[owner] += slice_size;
        }
    }

    let send_count_i32: Vec<i32> = send_count.iter().map(|&x| x as i32).collect();
    let send_displs_i32: Vec<i32> = send_displs.iter().map(|&x| x as i32).collect();
    let mut recv_count_i32 = vec![0i32; P];
    grid.cart_comm.all_to_all_into(&send_count_i32[..], &mut recv_count_i32[..]);

    let recv_count: Vec<usize> = recv_count_i32.iter().map(|&x| x as usize).collect();
    let mut recv_displs: Vec<usize> = vec![0; P];
    for i in 1..P {
        recv_displs[i] = recv_displs[i - 1] + recv_count[i - 1];
    }
    let recv_displs_i32: Vec<i32> = recv_displs.iter().map(|&x| x as i32).collect();
    let recv_total = recv_displs.last().unwrap() + recv_count.last().unwrap();
    let mut recv_buffer: Vec<f64> = vec![0.0; recv_total];

    let send_part = Partition::new(
        &send_buffer[..],
        &send_count_i32[..],
        &send_displs_i32[..],
    );

    let mut recv_part = PartitionMut::new(
        &mut recv_buffer[..],
        &recv_count_i32[..],
        &recv_displs_i32[..],
    );

    grid.cart_comm.all_to_all_varcount_into(&send_part, &mut recv_part);

    let mut new_data: Vec<f64> = vec![0.0; n0_global * n1_global * total_idx_recv];
    let mut local_n2_idx: Vec<usize> =  Vec::with_capacity(total_idx_recv);
    let mut recv_ptr: Vec<usize> = recv_displs.clone();

    for &blk in my_owned_blocks {
        let cur_block_idx_start = block_start_idx[blk];
        let cur_block_idx_end = if blk == num_block - 1 {n2_global - 1} else {block_start_idx[blk + 1] - 1};
        for cur_n2_idx in cur_block_idx_start..=cur_block_idx_end {
            local_n2_idx.push(cur_n2_idx);
            let slice_start = (local_n2_idx.len() - 1) * n0_global * n1_global;
            let mut slice_full = &mut new_data[slice_start..slice_start + n0_global * n1_global];
            for src in 0..P {
                let src_n0_size = all_local_n0[src];
                let src_n0_start = all_start_n0[src];
                let chunk = &recv_buffer[recv_ptr[src]..recv_ptr[src] + src_n0_size * n1_global];
                for j in 0..n1_global {
                    let from = &chunk[j * src_n0_size..(j+1) * src_n0_size];
                    let to_start = j * n0_global + src_n0_start;
                    slice_full[to_start..to_start + src_n0_size].copy_from_slice(from);
                }
                recv_ptr[src] += src_n0_size * n1_global;
            }
        }
    }

    let result = RIFull {
        size: [n0_global, n1_global, local_n2_idx.len()],
        indicing: [1, n0_global, n0_global * n1_global],
        data: new_data,
        dist_axis: 2,
        local_n2_idx: Some(local_n2_idx),
    };

    result
}

fn validate_redistribution(grid: &MPIGrid, original: &RIFull<f64>, redistributed: &RIFull<f64>) -> bool {
    assert!(original.dist_axis == 0);
    assert!(redistributed.dist_axis == 2);
    let P = grid.cart_comm.size() as usize;
    let my_rank = grid.rank as usize;
    let n0_global = redistributed.size[0];
    let n0_local = original.size[0];
    let n1_global = original.size[1];
    let n2_global = original.size[2];

    let mut local_sums: Vec<f64> = vec![0.0; n2_global];
    let slice_size = n0_local * n1_global;
    for (i, sum) in local_sums.iter_mut().enumerate() {
        let start = i * slice_size;
        let data = &original.data[start..start + slice_size];
        *sum = data.iter().map(|&x| x.abs()).sum::<f64>();
    }

    let mut global_sums: Vec<f64> = vec![0.0; n2_global];
    grid.cart_comm.all_reduce_into(&local_sums, &mut global_sums, &SystemOperation::sum());
    let owned_idx = redistributed
            .local_n2_idx
            .as_ref()
            .expect("Redistributed tensor must have local_n2_idx");
    let mut flag = true;

    for (pos, &idx) in owned_idx.iter().enumerate() {
        let slice_start = pos * n0_global * n1_global;
        let slice_data = &redistributed.data[slice_start..slice_start + n0_global * n1_global];
        let my_sum: f64 = slice_data.iter().map(|&x| x.abs()).sum::<f64>();
        let ref_sum = global_sums[idx];

        if (my_sum - ref_sum).abs() >= 1e-12 * ref_sum.abs().max(1.0) {
            eprintln!(
                "[Rank {}] Mismatch for n2 index {}: local sum = {}, reference = {}",
                    grid.rank, idx, my_sum, ref_sum
                );
                flag = false;
            }
    }
    let mut global_ok = false;
    grid.cart_comm.all_reduce_into(&flag, &mut global_ok, &SystemOperation::logical_and());
    global_ok
}

fn final_assembly(grid: &MPIGrid, row_buffer: &mut Vec<f64>, col_buffer: &mut Vec<f64>, ctx: &Ctx25dBlock, n0_global: usize, n1_global:usize, n2_global: usize) -> RIFull<f64> {
    let num_block = ctx.block_start_idx.len();
    let local_block_idx = &ctx.local_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let block_idx_sent_row = &ctx.block_idx_sent_row;
    let block_idx_sent_col = &ctx.block_idx_sent_col;
    let slice_size = n0_global * n1_global;
    let mut tensor_data_size: usize = 0;
    let mut local_block_size: Vec<usize> = Vec::new();
    for &blk in local_block_idx {
        let cur_block_size = if blk == num_block - 1 {n2_global - block_start_idx[blk]} else {block_start_idx[blk + 1] - block_start_idx[blk]};
        local_block_size.push(cur_block_size * slice_size);
        tensor_data_size += cur_block_size * slice_size;
    }
    let mut tensor_data: Vec<f64> = vec![0.0; tensor_data_size];

    let row_comm_size = grid.row_comm.size() as usize;
    assert_eq!(row_comm_size, grid.dims.1 as usize);
    let col_comm_size = grid.col_comm.size() as usize;
    assert_eq!(col_comm_size, grid.dims.0 as usize);

    let mut row_buffer_size: Vec<usize> = vec![0; row_comm_size];
    let mut row_buffer_displs: Vec<usize> = vec![0; row_comm_size];
    let mut col_buffer_size: Vec<usize> = vec![0; col_comm_size];
    let mut col_buffer_displs: Vec<usize> = vec![0; col_comm_size];

    for i in 0..row_comm_size {
        let cur_recv_idx = &block_idx_sent_row[i];
        let mut total_size: usize = 0;
        for &blk in cur_recv_idx {
            let pos = local_block_idx.iter().position(|&x| x == blk).expect("block not found in local_block_idx");
            total_size += local_block_size[pos];
        }
        row_buffer_size[i] = total_size;
    }
    for i in 1..row_comm_size {
        row_buffer_displs[i] = row_buffer_displs[i - 1] + row_buffer_size[i - 1];
    }

    for i in 0..col_comm_size {
        let cur_recv_idx = &block_idx_sent_col[i];
        let mut total_size: usize = 0;
        for &blk in cur_recv_idx {
            let pos = local_block_idx.iter().position(|&x| x == blk).expect("block not found in local_block_idx");
            total_size += local_block_size[pos];
        }
        col_buffer_size[i] = total_size;
    }
    for i in 1..col_comm_size {
        col_buffer_displs[i] = col_buffer_displs[i - 1] + col_buffer_size[i - 1];
    }

    assert_eq!(row_buffer.len(), row_buffer_size.last().unwrap() + row_buffer_displs.last().unwrap());
    assert_eq!(col_buffer.len(), col_buffer_size.last().unwrap() + col_buffer_displs.last().unwrap());

    let mut dest_offset: usize = 0;
    for i in 0..local_block_idx.len() {
        let cur_block_idx = local_block_idx[i];
        let cur_block_size = local_block_size[i];
        let mut src_j: isize = -1;
        let mut src_i: isize = -1;
        for j in 0..row_comm_size {
            for &blk in &block_idx_sent_row[j] {
                if blk == cur_block_idx {
                    src_j = j as isize;
                    let start = row_buffer_displs[j];
                    let end = row_buffer_displs[j] + cur_block_size;
                    tensor_data[dest_offset..dest_offset + cur_block_size].copy_from_slice(&row_buffer[start..end]);
                    row_buffer_displs[j] += cur_block_size;
                    break;
                }
            }
            if src_j != -1 {
                break;
            }
        }
        if src_j == -1 {
            for j in 0..col_comm_size {
                for &blk in &block_idx_sent_col[j] {
                    if blk == cur_block_idx {
                        src_i = j as isize;
                        assert!(j != grid.my_coords.0 as usize, "Column-only block from same row?");
                        let start = col_buffer_displs[j];
                        let end = col_buffer_displs[j] + cur_block_size;
                        tensor_data[dest_offset..dest_offset + cur_block_size].copy_from_slice(&col_buffer[start..end]);
                        col_buffer_displs[j] += cur_block_size;
                        break;
                    }
                }
            }
            assert!(src_i != -1, "Block {} not found in neither buffers", cur_block_idx);
        }
        dest_offset += cur_block_size;
    }
    row_buffer.clear();
    col_buffer.clear();

    let mut local_n2_idx: Vec<usize> = Vec::new();
    for &blk in local_block_idx {
        let cur_block_idx_start = block_start_idx[blk];
        let cur_block_idx_end = if blk == num_block - 1 {n2_global - 1} else {block_start_idx[blk + 1] - 1};
        for cur_n2_idx in cur_block_idx_start..=cur_block_idx_end {
            local_n2_idx.push(cur_n2_idx);
        }
    }

    let result = RIFull {
        size: [n0_global, n1_global, local_n2_idx.len()],
        indicing: [1, n0_global, n0_global * n1_global],
        data: tensor_data,
        dist_axis: 2,
        local_n2_idx: Some(local_n2_idx),
    };

    result

}

fn broadcast_by_axis(axis_comm: &SimpleCommunicator, dten: &RIFull<f64>) -> Vec<f64> {
    assert!(dten.dist_axis == 2);
    let axis_P = axis_comm.size() as usize;
    let send_count = dten.data.len();

    let mut recv_count: Vec<usize> = vec![0; axis_P];
    axis_comm.all_gather_into(&send_count, &mut recv_count[..]);
    let mut recv_displs: Vec<usize> = vec![0; axis_P];
    for i in 1..axis_P {
        recv_displs[i] = recv_displs[i - 1] + recv_count[i - 1];
    }

    let recv_total = recv_displs.last().unwrap() + recv_count.last().unwrap();
    let mut recv_buffer: Vec<f64> = vec![0.0; recv_total];
    let recv_count_i32: Vec<i32> = recv_count.iter().map(|&x| x as i32).collect();
    let recv_displs_i32: Vec<i32> = recv_displs.iter().map(|&x| x as i32).collect();
    let mut recv_part = PartitionMut::new(
            &mut recv_buffer[..],
            &recv_count_i32[..],
            &recv_displs_i32[..],
        );
    axis_comm.all_gather_varcount_into(&dten.data[..], &mut recv_part);
    recv_buffer
}

fn compute_double_gap(i_state_eigen: f64, j_state_eigen: f64, i_state_occ: f64, j_state_occ: f64,
    i_virt_eigen: f64, j_virt_eigen: f64, i_virt_occ: f64, j_virt_occ: f64) -> f64 {
    let ij_state_eigen = i_state_eigen + j_state_eigen;
    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
    let mut double_gap = (ij_virt_eigen - ij_state_eigen);
    if double_gap.abs()<=1.0E-6 {
        println!("Warning: too close to degeneracy");
        double_gap = 1.0e-6;
    };
    double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));
    double_gap
}

fn compute_eri_virt<'a, M>(eri_virt: &M, eigenvalues: &Vec<f64>, occupation: &Vec<f64>,
    i_state_eigen: f64, j_state_eigen: f64, i_state_occ: f64, j_state_occ: f64,
    virt_offset: usize, off_diag: bool) -> (f64, f64)
where M: BasicMatrix<'a, f64> + ?Sized
{
    assert_eq!(eri_virt.size()[0], eri_virt.size()[1], "eri_virt must be square");
    let n1 = eri_virt.size()[0];
    let data = eri_virt.data_ref().unwrap();

    let mut term_os = 0.0_f64;
    let mut term_ss = 0.0_f64;

    for a in 0..n1 {
        let i_virt_eigen = eigenvalues.get(a + virt_offset).unwrap().clone();
        let i_virt_occ = occupation.get(a + virt_offset).unwrap()/2.0.clone();
        for b in 0..n1 {
            let j_virt_eigen = eigenvalues.get(b + virt_offset).unwrap().clone();
            let j_virt_occ = occupation.get(b + virt_offset).unwrap()/2.0.clone();
            let double_gap = compute_double_gap(i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, i_virt_eigen, j_virt_eigen, i_virt_occ, j_virt_occ);
            let e_ab = data[a + b * n1];
            let e_ba = data[b + a * n1];
            term_os += e_ab * e_ab / double_gap;
            term_ss += (e_ab - e_ba) * e_ab / double_gap;
        }
    }

    if off_diag {
        term_os *= 2.0;
        term_ss *= 2.0;
    }

    (term_os, term_ss)

}

pub fn local_computation(final_tensor: &RIFull<f64>, ctx: &Ctx25dBlock, n2_global: usize,
    eigenvalues: &Vec<f64>, occupation: &Vec<f64>, occ_offset: usize, virt_offset: usize) -> (f64, f64) {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = final_tensor.size[0];
    let n1_global = final_tensor.size[1];
    let mut eri_virt: MatrixFull<f64> = MatrixFull::new([n1_global, n1_global], 0.0_f64);
    let mut local_term_os: f64 = 0.0_f64;
    let mut local_term_ss: f64 = 0.0_f64;

    for &row_blk in row_block_idx {
        let row_start = block_start_idx[row_blk];
        let row_end = if row_blk == num_block - 1 {n2_global - 1} else {block_start_idx[row_blk + 1] - 1};
        for &col_blk in col_block_idx {
            let col_start = block_start_idx[col_blk];
            let col_end = if col_blk == num_block - 1 {n2_global - 1} else {block_start_idx[col_blk + 1] - 1};
            if row_blk > col_blk {
                continue;
            }
            else if row_blk < col_blk {
                assert!(row_end < col_start);
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues.get(i + occ_offset).unwrap().clone();
                    let i_state_occ = occupation.get(i + occ_offset).unwrap()/2.0.clone();
                    for j in col_start..=col_end {
                        let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                        let j_state_eigen = eigenvalues.get(j + occ_offset).unwrap().clone();
                        let j_state_occ = occupation.get(j + occ_offset).unwrap()/2.0.clone();
                        _dgemm(&ri_i, (0..n0_global, 0..n1_global), 'T',
                               &ri_j, (0..n0_global, 0..n1_global), 'N',
                               &mut eri_virt, (0..n1_global, 0..n1_global),
                               1.0, 0.0);
                        let (term_os, term_ss) = compute_eri_virt(&eri_virt, eigenvalues, occupation, i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, virt_offset, true);
                        local_term_os -= term_os;
                        local_term_ss -= term_ss;
                    }
                }

            }
            else {
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues.get(i + occ_offset).unwrap().clone();
                    let i_state_occ = occupation.get(i + occ_offset).unwrap()/2.0.clone();
                    for j in col_start..=col_end {
                        if i <= j {
                            let off_diag = i != j;
                            let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                            let j_state_eigen = eigenvalues.get(j + occ_offset).unwrap().clone();
                            let j_state_occ = occupation.get(j + occ_offset).unwrap()/2.0.clone();
                            _dgemm(&ri_i, (0..n0_global, 0..n1_global), 'T',
                                   &ri_j, (0..n0_global, 0..n1_global), 'N',
                                   &mut eri_virt, (0..n1_global, 0..n1_global),
                                   1.0, 0.0);
                            let (term_os, term_ss) = compute_eri_virt(&eri_virt, eigenvalues, occupation, i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, virt_offset, off_diag);
                            local_term_os -= term_os;
                            local_term_ss -= term_ss;
                        }
                    }
                }
            }
        }
    }
    (local_term_os, local_term_ss)
}


pub fn local_computation_batch(final_tensor: &RIFull<f64>, ctx: &Ctx25dBlock, n2_global: usize,
 eigenvalues: &Vec<f64>, occupation: &Vec<f64>, occ_offset: usize, virt_offset: usize) -> (f64, f64) {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = final_tensor.size[0];
    let n1_global = final_tensor.size[1];
    let mut local_term_os: f64 = 0.0_f64;
    let mut local_term_ss: f64 = 0.0_f64;
    let mut size_buf = [0usize; 2];

    for &col_blk in col_block_idx {
        let col_start = block_start_idx[col_blk];
        let col_end = if col_blk == num_block - 1 {n2_global - 1} else {block_start_idx[col_blk + 1] - 1};
        let ri_j_block = final_tensor.get_reducing_block_matrix_global_n2(col_start, col_end, &mut size_buf).unwrap();
        let n1_block = ri_j_block.size[1];
        assert_eq!(n1_block % n1_global, 0);
        let bj = n1_block / n1_global;
        assert_eq!(bj, col_end - col_start + 1);

        let col_eigenvalues = &eigenvalues[col_start + occ_offset..col_end + occ_offset + 1];
        let col_occupation = &occupation[col_start + occ_offset..col_end + occ_offset + 1];

        let mut eri_virt: MatrixFull<f64> = MatrixFull::new([n1_global, n1_block], 0.0_f64);

        for &row_blk in row_block_idx {
            let row_start = block_start_idx[row_blk];
            let row_end = if row_blk == num_block - 1 {n2_global - 1} else {block_start_idx[row_blk + 1] - 1};
            if row_blk > col_blk {
                continue;
            }
            else if row_blk < col_blk {
                assert!(row_end < col_start);
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues.get(i + occ_offset).unwrap().clone();
                    let i_state_occ = occupation.get(i + occ_offset).unwrap()/2.0.clone();

                    _dgemm(&ri_i, (0..n0_global, 0..n1_global), 'T',
                           &ri_j_block, (0..n0_global, 0..n1_block), 'N',
                           &mut eri_virt, (0..n1_global, 0..n1_block),
                           1.0, 0.0);
                    let sub_size = [n1_global, n1_global];
                    let sub_indicing = [1usize, n1_global];
                    let terms: Vec<(f64, f64)> = (0..bj).into_par_iter().map(|j_offset| {
                        let start = j_offset * n1_global * n1_global;
                        let end = start + n1_global * n1_global;
                        let sub_view = MatrixFullSlice {
                            size: &sub_size[..],
                            indicing: &sub_indicing[..],
                            data: &eri_virt.data[start..end],
                        };
                        let j_state_eigen = col_eigenvalues[j_offset].clone();
                        let j_state_occ = col_occupation[j_offset] / 2.0.clone();
                        compute_eri_virt(&sub_view, eigenvalues, occupation, i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, virt_offset, true)
                    }).collect();
                    for (term_os, term_ss) in terms {
                        local_term_os -= term_os;
                        local_term_ss -= term_ss;
                    }

                }

            }
            else {
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues.get(i + occ_offset).unwrap().clone();
                    let i_state_occ = occupation.get(i + occ_offset).unwrap()/2.0.clone();
                    _dgemm(&ri_i, (0..n0_global, 0..n1_global), 'T',
                           &ri_j_block, (0..n0_global, 0..n1_block), 'N',
                           &mut eri_virt, (0..n1_global, 0..n1_block),
                           1.0, 0.0);

                    let sub_size = [n1_global, n1_global];
                    let sub_indicing = [1usize, n1_global];
                    let i_offset = i - row_start;

                    let terms: Vec<(f64, f64)> = (i_offset..bj).into_par_iter().map(|j_offset| {
                        let start = j_offset * n1_global * n1_global;
                        let end = start + n1_global * n1_global;
                        let sub_view = MatrixFullSlice {
                            size: &sub_size[..],
                            indicing: &sub_indicing[..],
                            data: &eri_virt.data[start..end],
                        };
                        let off_diag = i_offset != j_offset;
                        let j_state_eigen = col_eigenvalues[j_offset].clone();
                        let j_state_occ = col_occupation[j_offset] / 2.0.clone();
                        compute_eri_virt(&sub_view, eigenvalues, occupation, i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, virt_offset, off_diag)
                    }).collect();
                    for (term_os, term_ss) in terms {
                        local_term_os -= term_os;
                        local_term_ss -= term_ss;
                    }
                }

            }
        }
    }
    (local_term_os, local_term_ss)
}


pub fn swap_ownership(grid: &MPIGrid, ctx: &Ctx25dBlock, local_tensor: &RIFull<f64>,
    n0_global: usize, n1_global: usize, n2_global: usize, local_n0_range: &std::ops::Range<usize>) -> RIFull<f64> {
    let diag_tensor = redistribute_to_diag(&grid, &ctx, &local_tensor, n0_global, &local_n0_range);
    //let flag = validate_redistribution(&grid, &local_tensor, &diag_tensor);
    //assert!(flag, "redistribution fails");
    let mut row_buffer = broadcast_by_axis(&grid.row_comm, &diag_tensor);
    let mut col_buffer = broadcast_by_axis(&grid.col_comm, &diag_tensor);
    drop(diag_tensor);
    let final_tensor = final_assembly(&grid, &mut row_buffer, &mut col_buffer, &ctx, n0_global, n1_global, n2_global);
    //let final_flag = validate_redistribution(&grid, &local_tensor, &final_tensor);
    //assert!(final_flag, "final assembly fails");
    final_tensor
}
