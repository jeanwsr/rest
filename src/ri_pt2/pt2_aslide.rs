use mpi::ffi;
use crate::mpi_io::*;
use rest_tensors::ri::*;
use mpi::collective::SystemOperation;

use mpi::traits::*;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::{SimpleCommunicator};

use std::mem;
use std::env;
use std::ops::Range;
use std::collections::{HashMap, HashSet};
use std::time::{Instant, Duration};

use rest_tensors::RIFull;
use rest_tensors::{TensorOpt, MatrixFull, MatrixFullSlice, BasicMatrix};
use rest_tensors::matrix_blas_lapack::{_dgemm};
use libc::{sysconf, _SC_PHYS_PAGES};
use crate::utilities::memory_batch::{detect_available_memory_mb};


use std::ptr;
use std::thread;
use libc::c_void;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;


#[derive(Debug, Clone)]
pub struct State {
    pub to_compute: Vec<(usize, usize)>,  // block pairs (I,J) in this tile
    pub to_fetch: Vec<usize>,             // blocks to load before tile
    pub to_release: Vec<usize>,           // blocks to evict after previous tile
}

pub struct CtxASlideBlock {
    pub k: usize,
    pub block_start_idx: Vec<usize>,
    pub ownership: Vec<Vec<usize>>,
    pub row_block_idx: Vec<usize>,
    pub col_block_idx: Vec<usize>,
    pub schedule: Vec<State>,
    pub kernel_size: (usize, usize),
    pub cache_capacity: usize,
}


pub fn generate_schedule(
    row_blocks: &[usize],
    col_blocks: &[usize],
    kernel_row: usize,
    kernel_col: usize,
    row_offset: usize,
    col_offset: usize,
    cache_capacity: usize,
    upper_only: bool,
) -> Vec<State> {
    let R = row_blocks.len();
    let C = col_blocks.len();

    let kernel_row = kernel_row.max(1);
    let kernel_col = kernel_col.max(1);

    let num_tile_rows = (R + kernel_row - 1) / kernel_row;
    let num_tile_cols = (C + kernel_col - 1) / kernel_col;

    // Tile lists with offsets (rotate tile indices)
    let mut tile_rows: Vec<usize> = (0..num_tile_rows).collect();
    let mut tile_cols: Vec<usize> = (0..num_tile_cols).collect();
    if row_offset > 0 { tile_rows.rotate_left(row_offset % num_tile_rows); }
    if col_offset > 0 { tile_cols.rotate_left(col_offset % num_tile_cols); }

    // Build pair_order and per-tile block usage
    let mut tmp_to_compute: Vec<(usize, usize)> = Vec::new();
    let mut step_count: Vec<usize> = Vec::new();          // end offset of each non-empty tile
    let mut tile_blocks: Vec<Vec<usize>> = Vec::new();    // blocks used in each tile

    let mut forward = true;
    for &tr in &tile_rows {
        let r_start = tr * kernel_row;
        let r_end = (r_start + kernel_row).min(R);
        let mut col_order = tile_cols.clone();
        if !forward { col_order.reverse(); }

        for &tc in &col_order {
            let c_start = tc * kernel_col;
            let c_end = (c_start + kernel_col).min(C);

            let mut cur_pairs = Vec::new();
            let mut cur_blocks = HashSet::new();
            for i in r_start..r_end {
                let I = row_blocks[i];
                for j in c_start..c_end {
                    let J = col_blocks[j];
                    if (upper_only && I <= J) || (!upper_only) {
                        cur_pairs.push((I, J));
                        cur_blocks.insert(I);
                        cur_blocks.insert(J);
                    }
                }
            }
            if !cur_pairs.is_empty() {
                tmp_to_compute.extend(cur_pairs);
                step_count.push(tmp_to_compute.len());
                tile_blocks.push(cur_blocks.into_iter().collect());
            }
        }
        forward = !forward;
    }

    let num_steps = step_count.len();

    // Use map: block -> list of step indices where used
    let mut use_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (step, blocks) in tile_blocks.iter().enumerate() {
        for &blk in blocks {
            use_map.entry(blk).or_default().push(step);
        }
    }

    let find_next_use = |blk: usize, cur_step: usize| -> usize {
        use_map.get(&blk)
            .and_then(|steps| steps.iter().copied().find(|&s| s > cur_step))
            .unwrap_or(usize::MAX)
    };

    // Determine initial prefetch: first `cache_capacity` blocks in order of first use.
    let mut initial_prefetch: Vec<usize> = Vec::new();

    let mut first_uses: Vec<(usize, usize)> = Vec::new();  // (first_use_step, block_id)
    for (&blk, steps) in use_map.iter() {
        if let Some(&first) = steps.first() {
            first_uses.push((first, blk));
        }
    }
    first_uses.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    for (_, blk) in first_uses {
        if initial_prefetch.len() < cache_capacity {
            initial_prefetch.push(blk);
        } else {
            break;
        }
    }

    // Prefetch state

    let mut schedule: Vec<State> = Vec::with_capacity(num_steps + 1);
    let mut cache: HashSet<usize> = initial_prefetch.iter().copied().collect();
    schedule.push(State {
        to_compute: Vec::new(),
        to_fetch: initial_prefetch.clone(),
        to_release: Vec::new(),
    });

    let mut step_start = 0;
    for step in 0..num_steps {
        let step_end = step_count[step];
        let to_compute: Vec<(usize, usize)> = tmp_to_compute[step_start..step_end].to_vec();
        step_start = step_end;

        let cur_blocks = &tile_blocks[step];
        let mut to_fetch = Vec::new();
        for &blk in cur_blocks {
            if !cache.contains(&blk) {
                to_fetch.push(blk);
                cache.insert(blk);
            }
        }

        let mut to_release = Vec::new();
        while cache.len() > cache_capacity {
            // evict block with farthest next use among those not needed now
            let mut evict = None;
            let mut farthest = 0;
            for &blk in &cache {
                if cur_blocks.contains(&blk) { continue; }
                let next = find_next_use(blk, step);
                if next > farthest {
                    farthest = next;
                    evict = Some(blk);
                }
            }
            if let Some(blk) = evict {
                cache.remove(&blk);
                to_release.push(blk);
            } else {
                break;
            }
        }

        schedule.push(State { to_compute, to_fetch, to_release });
    }

    schedule
}

pub fn validate_schedule(schedule: &Vec<State>, cache_capacity: usize, mpi_op: &MPIOperator) -> bool
{
    let num_sliding_steps = schedule.len();
    let mut cache: HashSet<usize> = HashSet::new();
    let mut local_flag: bool = true;
    for step in 0..num_sliding_steps {
        let cur_state: &State = &schedule[step];
        let to_compute: &Vec<(usize, usize)> = &cur_state.to_compute;
        let to_fetch: &Vec<usize> = &cur_state.to_fetch;
        let to_release: &Vec<usize> = &cur_state.to_release;

        for &blk in to_release {
            if (!cache.contains(&blk)) {
                println!("block to be released not found.");
                local_flag = false;
                break;
            }
            cache.remove(&blk);
        }

        for &blk in to_fetch {
            cache.insert(blk);
        }

        if cache.len() > cache_capacity {
            println!("stack overflow:({},{}) at step:{}.", cache.len(), cache_capacity, step);
            local_flag = false;
            break;
        }

        for &(I,J) in to_compute {
            if (!cache.contains(&I) || !cache.contains(&J)) {
                println!("compute block not found.");
                local_flag = false;
                break;
            }
        }

        if !local_flag {
            break;
        }

    }
    let mut global_flag = false;
    mpi_op.world.all_reduce_into(&local_flag, &mut global_flag, &SystemOperation::logical_and());
    global_flag
}

pub fn initialize_metadata_ASlide(mpi_op: &MPIOperator, n0_global:usize, n1_global: usize, n2_global: usize, close_shell: bool) -> CtxASlideBlock {
    let size = mpi_op.size;
    let rank = mpi_op.rank;
    let (r, c) = {
        let mut r = (size as f64).sqrt() as i32;
        let size_i32 = size as i32;
        while size_i32 % r != 0 {
            r -= 1;
        }
        (r, size_i32 / r)
    };
    let r = r as usize;
    let c = c as usize;
    let p = r * c;
    let (my_row, my_col) = (rank / c, rank % c);
    //println!("rank: {} -> ({},{})", rank, my_row, my_col);

    let grid_mult = if r == c {r} else {p};
    let target_block_size: usize = 4;
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

    let mut ownership: Vec<Vec<usize>> = vec![Vec::new(); p];
    let mut row_block_idx = Vec::new();
    let mut col_block_idx = Vec::new();

    for blk in 0..num_block {
        let i = blk % r;
        let j = blk % c;
        if i == my_row {
            row_block_idx.push(blk);
        }
        if j == my_col {
            col_block_idx.push(blk);
        }
        let owner = blk % p;
        ownership[owner].push(blk);
    }
    let max_block_size = if block_residue > 0 { block_size + 1 } else { block_size };
    let cache_capacity: usize = determine_cache_capacity(mpi_op, n0_global * n1_global, max_block_size);
    assert!(cache_capacity > 1);
    let block_bandwidth = determine_bandwidth(mpi_op, n0_global * n1_global, max_block_size);
    let block_dgemm_speed = determine_dgemm_speed(mpi_op, n0_global, n1_global, max_block_size);
    let (kernel_row, kernel_col) = determine_kernel_size(cache_capacity, block_bandwidth, block_dgemm_speed);

    let schedule = generate_schedule(&row_block_idx, &col_block_idx, kernel_row, kernel_col, my_row, my_col, cache_capacity, close_shell);
    let schedule_flag = validate_schedule(&schedule, cache_capacity, mpi_op);
    assert!(schedule_flag, "schedule failed!");

    if (rank == 0) {
        println!("k:{}, num_blocks:{}, block_size:{}", k, num_block, max_block_size);
        println!("cache_capacity:{} blocks", cache_capacity);
        println!("FLOPS/COMM = {}", (block_dgemm_speed as f64 / block_bandwidth as f64));
        println!("kernel size:({},{})", kernel_row, kernel_col);
    }
    eprintln!("rank:{} schedule steps:{}", rank, schedule.len());

    CtxASlideBlock {
        k,
        block_start_idx,
        ownership,
        row_block_idx,
        col_block_idx,
        schedule,
        kernel_size: (kernel_row, kernel_col),
        cache_capacity: cache_capacity,
    }
}

pub fn determine_cache_capacity(mpi_op: &MPIOperator, slice_size: usize, block_size: usize) -> usize {
    let avail_mb = detect_available_memory_mb();
    let elem_size = std::mem::size_of::<f64>();
    let block_mb: f64 = slice_size as f64 * block_size as f64 * elem_size as f64 / 1024.0 / 1024.0;
    let local_capacity : usize = (avail_mb / block_mb).floor() as usize;
    let mut global_capacity: usize = 0;
    println!("rank:{}, avail_mb:{}, block_mb:{}, local_capacity:{}", mpi_op.rank, avail_mb, block_mb, local_capacity);
    mpi_op.world.all_reduce_into(&local_capacity, &mut global_capacity, SystemOperation::min());
    global_capacity
}

pub fn determine_dgemm_speed(mpi_op: &MPIOperator, n0: usize, n1: usize, block_size: usize) -> f64 {
    let mut a = MatrixFull::new([n0, n1], 1.0_f64);
    let mut b = MatrixFull::new([n0, block_size * n1], 1.0_f64);
    let mut c = MatrixFull::new([n1, block_size * n1], 0.0_f64);
    _dgemm(
            &a, (0..n0, 0..n1), 'T',
            &b, (0..n0, 0..block_size * n1), 'N',
            &mut c, (0..n1, 0..block_size * n1),
            1.0, 0.0,
    );
    const ITERATIONS: usize = 5;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        _dgemm(
            &a, (0..n0, 0..n1), 'T',
            &b, (0..n0, 0..block_size * n1), 'N',
            &mut c, (0..n1, 0..block_size * n1),
            1.0, 0.0,
        );
    }
    let elapsed = start.elapsed().as_secs_f64() / (ITERATIONS as f64);
    let mut total_elapsed = 0.0_f64;
    mpi_op.world.all_reduce_into(&elapsed, &mut total_elapsed, SystemOperation::sum());
    let result = total_elapsed * (block_size as f64) / (mpi_op.size as f64);
    elapsed
}

pub fn determine_bandwidth(mpi_op: &MPIOperator, slice_size: usize, block_size: usize) -> f64 {
    let rank = mpi_op.rank;
    let size = mpi_op.size;
    assert!(size > 1);
    let n_doubles = slice_size * block_size;
    let send_buf = vec![0.0_f64; n_doubles];
    let mut recv_buf = vec![0.0_f64; n_doubles];

    let mut one_way = 0.0;

    // Only ranks 0 and 1 perform the ping‑pong.
    if rank == 0 || rank == 1 {
        // Warm‑up round
        if rank == 0 {
            mpi_op.world.process_at_rank(1).send(&send_buf[..]);
            mpi_op.world.process_at_rank(1).receive_into(&mut recv_buf[..]);
        } else {
            mpi_op.world.process_at_rank(0).receive_into(&mut recv_buf[..]);
            mpi_op.world.process_at_rank(0).send(&send_buf[..]);
        }

        const ITERATIONS: usize = 10;
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            if rank == 0 {
                mpi_op.world.process_at_rank(1).send(&send_buf[..]);
                mpi_op.world.process_at_rank(1).receive_into(&mut recv_buf[..]);
            } else {
                mpi_op.world.process_at_rank(0).receive_into(&mut recv_buf[..]);
                mpi_op.world.process_at_rank(0).send(&send_buf[..]);
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        one_way = elapsed / (ITERATIONS as f64) / 2.0;
    }

    // Broadcast the measured one‑way time from rank 0 to all ranks
    let mut result = vec![one_way];
    mpi_op.world.process_at_rank(0).broadcast_into(&mut result);
    result[0]
}

pub fn determine_kernel_size(cache_capacity: usize, block_bandwidth: f64, block_dgemm_speed: f64) -> (usize, usize)
{
    assert!(cache_capacity > 1);

    let target_min = 5.0;
    let target_max = 15.0;
    let target_mid = (target_min + target_max) / 2.0;

    let f2c_ratio = block_dgemm_speed / block_bandwidth;
    let ideal_kr = 2.0 * target_mid / f2c_ratio;
    let mut kr = ideal_kr.round() as usize;
    kr = kr.max(1);
    let mut kc = kr;

    if kr + kc > cache_capacity {
        let max_square = (cache_capacity / 2).max(1);
        kr = max_square;
        kc = max_square;
    }

    let mut candidates: Vec<(usize, usize)> = Vec::new();
        for dr in -2..=2 {
            for dc in -2..=2 {
                let r = kr as isize + dr;
                let c = kc as isize + dc;
                if r < 1 || c < 1 { continue; }
                let r = r as usize;
                let c = c as usize;
                if r + c <= cache_capacity {
                    candidates.push((r, c));
                }
            }
        }
    candidates.push((1, 1));

    let mut best: Option<(usize, usize, f64)> = None;
        for (r, c) in candidates {
            let ratio = (r as f64 * c as f64 * block_dgemm_speed)
                / ((r + c) as f64 * block_bandwidth);

            // Score: distance from the target range.
            let score = if ratio >= target_min && ratio <= target_max {
                (ratio - target_mid).abs()
            } else if ratio < target_min {
                target_min - ratio
            } else {
                ratio - target_max
            };

            if best.is_none() || score < best.as_ref().unwrap().2 {
                best = Some((r, c, score));
            }
    }

    best.map(|(r, c, _)| (r, c)).unwrap_or((1, 1))

}

pub fn redistribute_batch_reverse_w_truncation(mpi_op: &MPIOperator, ctx: &CtxASlideBlock, dten: &mut RIFull<f64>, n0_global: usize, n0_range: &Range<usize>)
-> HashMap<usize, RIFull<f64>>
{
    let my_rank = mpi_op.rank as usize;
    let P = mpi_op.size as usize;
    let block_start_idx = &ctx.block_start_idx;
    let num_blocks = block_start_idx.len();
    let n1_global = dten.size[1];
    let n2_global = dten.size[2];

    let start_n0 = n0_range.start;
    let local_n0 = n0_range.end - n0_range.start;

    // Gather n0 distribution info
    let mut all_start_n0 = vec![0usize; P];
    let mut all_local_n0 = vec![0usize; P];
    mpi_op.world.all_gather_into(&start_n0, &mut all_start_n0);
    mpi_op.world.all_gather_into(&local_n0, &mut all_local_n0);

    let slice_size_local = local_n0 * n1_global;   // doubles per slice piece (local n0)

    let mut result = HashMap::new();

    // Loop over global block IDs in batches (not just local owned blocks)
    let batch_size = (ctx.cache_capacity / 2).min(1);
    let mut batch_end = num_blocks;
    while batch_end > 0 {
        let mut batch_start = batch_end.saturating_sub(batch_size).max(0);
        let batch_blocks: Vec<usize> = (batch_start..batch_end).collect();

        // Build subset ownership for this batch: which blocks belong to which rank
        let mut subset_ownership: Vec<Vec<usize>> = vec![Vec::new(); P];
        for &blk in &batch_blocks {
            let owner = blk % P;   // uniform ownership
            subset_ownership[owner].push(blk);
        }

        // Compute send counts: for each destination, sum sizes of blocks it owns in this batch
        let mut send_count = vec![0usize; P];
        for dest in 0..P {
            let mut cnt = 0;
            for &blk in &subset_ownership[dest] {
                let start = block_start_idx[blk];
                let end = if blk == num_blocks - 1 { n2_global } else { block_start_idx[blk + 1] };
                cnt += slice_size_local * (end - start);
            }
            send_count[dest] = cnt;
        }

        // Build send buffer containing local pieces for all blocks in this batch
        let mut send_displs = vec![0usize; P];
        for i in 1..P { send_displs[i] = send_displs[i-1] + send_count[i-1]; }
        let send_total = send_displs.last().unwrap() + send_count.last().unwrap();
        let mut send_buffer = vec![0.0f64; send_total];
        let mut dest_ptr = send_displs.clone();

        // Pack data for every block in this batch
        for &blk in &batch_blocks {
            let owner = blk % P;
            let start = block_start_idx[blk];
            let end = if blk == num_blocks - 1 { n2_global } else { block_start_idx[blk + 1] };
            for n2 in start..end {
                let src_start = n2 * slice_size_local;
                let src = &dten.data[src_start..src_start + slice_size_local];
                let dst_start = dest_ptr[owner];
                send_buffer[dst_start..dst_start + slice_size_local].copy_from_slice(src);
                dest_ptr[owner] += slice_size_local;
            }
        }

        // Exchange data
        let send_count_i32: Vec<i32> = send_count.iter().map(|&x| x as i32).collect();
        let send_displs_i32: Vec<i32> = send_displs.iter().map(|&x| x as i32).collect();
        let mut recv_count_i32 = vec![0i32; P];
        mpi_op.world.all_to_all_into(&send_count_i32[..], &mut recv_count_i32[..]);

        let recv_count: Vec<usize> = recv_count_i32.iter().map(|&x| x as usize).collect();
        let mut recv_displs = vec![0usize; P];
        for i in 1..P { recv_displs[i] = recv_displs[i-1] + recv_count[i-1]; }
        let recv_total = recv_displs.last().unwrap() + recv_count.last().unwrap();
        let mut recv_buffer = vec![0.0f64; recv_total];

        let send_part = Partition::new(&send_buffer[..], &send_count_i32[..], &send_displs_i32[..]);
        let recv_displs_i32: Vec<i32> = recv_displs.iter().map(|&x| x as i32).collect();
        let mut recv_part = PartitionMut::new(&mut recv_buffer[..], &recv_count_i32[..], &recv_displs_i32[..]);
        mpi_op.world.all_to_all_varcount_into(&send_part, &mut recv_part);

        // Assemble only the blocks this rank owns (subset_ownership[my_rank])
        let mut recv_ptr = recv_displs.clone();
        for &blk in &subset_ownership[my_rank] {
            let start = block_start_idx[blk];
            let end = if blk == num_blocks - 1 { n2_global } else { block_start_idx[blk + 1] };
            let num_slices = end - start;

            let mut block_data = vec![0.0f64; num_slices * n0_global * n1_global];
            let mut block_n2_idx = Vec::with_capacity(num_slices);

            for n2 in start..end {
                block_n2_idx.push(n2);
                let slice_pos = (n2 - start) * n0_global * n1_global;
                let slice_full = &mut block_data[slice_pos..slice_pos + n0_global * n1_global];

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

            let block_tensor = RIFull {
                size: [n0_global, n1_global, num_slices],
                indicing: [1, n0_global, n0_global * n1_global],
                data: block_data,
                dist_axis: 2,
                local_n2_idx: Some(block_n2_idx),
            };
            result.insert(blk, block_tensor);
        }

        let new_n2_end = block_start_idx[batch_start];
        let new_data_len = new_n2_end * slice_size_local;
        dten.data.truncate(new_data_len);

        batch_end = batch_start;
    }

    result
}

pub fn computation_close_shell_slide(final_tensor: &HashMap<usize, RIFull<f64>>, ctx: &CtxASlideBlock, n2_global: usize,
 eigenvalues: &Vec<f64>, occupation: &Vec<f64>, occ_offset: usize, virt_offset: usize) -> (f64, f64) {
    let schedule = ctx.schedule;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();

    let mut local_term_os: f64 = 0.0_f64;
    let mut local_term_ss: f64 = 0.0_f64;
    let mut size_buf = [0usize; 2];
/*
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
                        compute_eri_virt_close_shell(&sub_view, eigenvalues, occupation, i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, virt_offset, true)
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
                        compute_eri_virt_close_shell(&sub_view, eigenvalues, occupation, i_state_eigen, j_state_eigen, i_state_occ, j_state_occ, virt_offset, off_diag)
                    }).collect();
                    for (term_os, term_ss) in terms {
                        local_term_os -= term_os;
                        local_term_ss -= term_ss;
                    }
                }

            }
        }
    }
    */
    (local_term_os, local_term_ss)
}



const TAG_REQUEST: i32 = 100;
const TAG_BLOCK:   i32 = 101;
pub struct ServicePortal<'a> {
    pub mpi_op: &'a MPIOperator,
    pub data: &'a HashMap<usize, RIFull<f64>>,
    pending_fetch_requests: Vec<ffi::MPI_Request>,   // Irecv + Isend for own fetches
    pending_data_sends: Vec<ffi::MPI_Request>,       // Isends we issued while serving
    done_request_started: bool,
    done_request: ffi::MPI_Request,
    local_done: i32,
    global_done: i32,
}

impl<'a> ServicePortal<'a> {
    pub fn new(mpi_op: &'a MPIOperator, data: &'a HashMap<usize, RIFull<f64>>) -> Self {
        ServicePortal
        {
            mpi_op: mpi_op,
            data: data,
            pending_fetch_requests: Vec::new(),
            pending_data_sends: Vec::new(),
            done_request_started: false,
            done_request: unsafe { mem::zeroed() },
            local_done: 0,
            global_done: 0,
        }
    }

    pub fn request_block(&mut self, block_id: usize, owner: i32, buffer: &mut [f64]) {
        let mut recv_req: ffi::MPI_Request = unsafe { mem::zeroed() };
        let mut send_req: ffi::MPI_Request = unsafe { mem::zeroed() };
        let blk = block_id as i32;
        unsafe {
            ffi::MPI_Irecv(
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len() as i32,
                ffi::RSMPI_DOUBLE,
                owner,
                TAG_BLOCK,
                self.mpi_op.world.as_raw(),
                &mut recv_req,
            );

            ffi::MPI_Isend(
                &blk as *const i32 as *const c_void,
                1,
                ffi::RSMPI_INT32_T,
                owner,
                TAG_REQUEST,
                self.mpi_op.world.as_raw(),
                &mut send_req,
            );
        }
        self.pending_fetch_requests.push(recv_req);
        self.pending_fetch_requests.push(send_req);

    }

    pub fn wait_for_fetches_with_service(&mut self) {
        while !self.pending_fetch_requests.is_empty() {
            // Handle incoming block requests from other ranks
            self.service();

            let mut flag: i32 = 0;
            unsafe {
                ffi::MPI_Testall(
                    self.pending_fetch_requests.len() as i32,
                    self.pending_fetch_requests.as_mut_ptr(),
                    &mut flag,
                    ffi::RSMPI_STATUSES_IGNORE,
                );
            }

            if flag != 0 {
                // Fetches are complete. Drain any requests that arrived just
                // before the test to ensure the queue is empty.
                self.service();
                self.pending_fetch_requests.clear();
                break;
            }

            // Optional: yield to avoid busy-waiting
            std::thread::yield_now();
        }
    }

    pub fn service(&mut self) {
        let mut flag: i32 = 0;
        let mut status: ffi::MPI_Status = unsafe { std::mem::zeroed() };

        loop {
            unsafe {
                ffi::MPI_Iprobe(
                    ffi::RSMPI_ANY_SOURCE,
                    TAG_REQUEST,
                    self.mpi_op.world.as_raw(),
                    &mut flag,
                    &mut status,
                );
            }
            if flag == 0 {
                break;
            }

            let mut block_id: i32 = 0;
            unsafe {
                ffi::MPI_Recv(
                    &mut block_id as *mut i32 as *mut c_void,
                    1,
                    ffi::RSMPI_INT32_T,
                    status.MPI_SOURCE,
                    TAG_REQUEST,
                    self.mpi_op.world.as_raw(),
                    ffi::RSMPI_STATUS_IGNORE,
                );
            }

            let blk = block_id as usize;
            let tensor = match self.data.get(&blk) {
                Some(t) => t,
                None => {
                    eprintln!(
                        "Rank {}: block {} requested but not owned locally.",
                        self.mpi_op.rank, blk
                    );
                    unsafe { ffi::MPI_Abort(self.mpi_op.world.as_raw(), 1) };
                    unreachable!();
                }
            };

            let data_ptr = tensor.data.as_ptr() as *const f64;
            let block_size = tensor.data.len() as i32;

            let mut sreq: ffi::MPI_Request = unsafe { std::mem::zeroed() };
            unsafe {
                ffi::MPI_Isend(
                    data_ptr as *const c_void,
                    block_size,
                    ffi::RSMPI_DOUBLE,
                    status.MPI_SOURCE,
                    TAG_BLOCK,
                    self.mpi_op.world.as_raw(),
                    &mut sreq,
                );
            }
            self.pending_data_sends.push(sreq);
        }

        // Progress non‑blocking allreduce if started
        if self.done_request_started {
            let mut done_flag: i32 = 0;
            unsafe {
                ffi::MPI_Test(
                    &mut self.done_request as *mut ffi::MPI_Request,
                    &mut done_flag,
                    ffi::RSMPI_STATUS_IGNORE,
                );
            }
        }
    }


    pub fn signal_done(&mut self) {
        if !self.done_request_started {
            self.local_done = 1;
            unsafe {
                ffi::MPI_Iallreduce(
                    &self.local_done as *const i32 as *const c_void,
                    &mut self.global_done as *mut i32 as *mut c_void,
                    1,
                    ffi::RSMPI_INT32_T,
                    ffi::RSMPI_SUM,
                    self.mpi_op.world.as_raw(),
                    &mut self.done_request,
                );
            }
            self.done_request_started = true;
        }
    }

    pub fn is_done(&mut self) -> bool {
        if !self.done_request_started {
            return false;
        }
        let mut flag: i32 = 0;
        unsafe {
            ffi::MPI_Test(
                &mut self.done_request as *mut ffi::MPI_Request,
                &mut flag,
                ffi::RSMPI_STATUS_IGNORE,
            );
        }
        flag != 0
    }

    pub fn finalize(&mut self) {
        if !self.pending_data_sends.is_empty() {
            unsafe {
                ffi::MPI_Waitall(
                    self.pending_data_sends.len() as i32,
                    self.pending_data_sends.as_mut_ptr(),
                    ffi::RSMPI_STATUSES_IGNORE,
                );
            }
            self.pending_data_sends.clear();
        }
    }
}
