//! Memory size detection and batching utilities.

use num_traits::ToPrimitive;

/// Detect available memory in system in MB.
pub fn detect_available_memory_mb() -> f64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory() as f64 / 1024.0 / 1024.0
}

/// Detect used memory in system in MB.
///
/// # Parameters
///
/// - `use_case`: `&str`
///
///   - `"sys"`: detect used memory of whole system.
///   - `"proc"`: detect used memory of current process.
pub fn detect_used_memory_mb(use_case: &str) -> f64 {
    match use_case {
        "sys" => {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            sys.used_memory() as f64 / 1024.0 / 1024.0
        },
        "proc" => {
            let mut sys = sysinfo::System::new();
            let pid = sysinfo::get_current_pid().unwrap();
            sys.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                true,
                sysinfo::ProcessRefreshKind::nothing().with_memory(),
            );
            let process = sys.process(pid).unwrap();
            process.memory() as f64 / 1024.0 / 1024.0
        },
        _ => {
            panic!("Unknown use_case for detect_used_memory_mb: {}", use_case);
        },
    }
}
/// Lower bound of the XC gradient block budget (MiB): a process that already exceeds its
/// declared budget still walks the grid in usable blocks.
pub const MIN_XC_GRAD_BLOCK_MB: f64 = 8.0;
/// Upper bound of the XC gradient block budget (MiB): one block never dominates the resident
/// set even when the declared budget is generous.
pub const MAX_XC_GRAD_BLOCK_MB: f64 = 256.0;

/// Working-set budget (MiB) of one XC gradient grid block, **per worker**.
///
/// `max_memory` is the declared process budget in MiB, the unit of `[ctrl] max_memory`.
/// The budget of one block of one worker is the head room left after the resident set, capped
/// at 10% of the declared budget (a single block must never claim the whole process budget) and
/// divided by the number of rayon workers of the grid loop. It is clamped to
/// [`MIN_XC_GRAD_BLOCK_MB`, `MAX_XC_GRAD_BLOCK_MB`].
///
/// `fallback_mb` is used when no budget is declared, which keeps the historical block size of
/// the gradient path.
///
/// `REST_XC_GRAD_BLK_MB` overrides the result, for experiments and for machines where the
/// declared budget does not describe the process.
pub fn xc_grad_block_mb(max_memory: Option<f64>, fallback_mb: f64) -> f64 {
    if let Ok(v) = std::env::var("REST_XC_GRAD_BLK_MB") {
        if let Ok(mb) = v.parse::<f64>() {
            return mb.max(1.0);
        }
    }
    match max_memory {
        Some(total) => {
            let avail = (total - detect_used_memory_mb("proc")).max(0.0);
            let workers = rayon::current_num_threads().max(1) as f64;
            (avail.min(0.1 * total) / workers).clamp(MIN_XC_GRAD_BLOCK_MB, MAX_XC_GRAD_BLOCK_MB)
        }
        None => fallback_mb,
    }
}

/// Peak-attribution probe: print the resident set of this process together with a label.
///
/// Enabled by the environment variable `REST_MEM_PROBE=1` or by `print_level >= 3`, and a no-op
/// otherwise, so the calls can stay in the hot paths of the force driver. Used to attribute the
/// peak memory of an analytic gradient.
pub fn mem_probe(label: &str, enabled: bool) {
    if !enabled {
        return;
    }
    println!("[mem-probe] {:9.1} MB | {}", detect_used_memory_mb("proc"), label);
}

/// Whether the memory probe is switched on for this run.
pub fn mem_probe_enabled(print_level: usize) -> bool {
    print_level >= 3 || std::env::var("REST_MEM_PROBE").map(|v| v == "1").unwrap_or(false)
}

/// Calculate batch size within possible memory.
///
/// For example, if we want to compute tensor (100, 100, 100), but only 50,000 memory available,
/// then this tensor should be splited into 20 batches.
///
/// ``flop`` in parameters is number of data, not refers to FLOPs.
///
/// This function requires generic `<T>`, which determines size of data.
///
/// # Parameters
///
/// - `unit_flop`: Number of data for unit operation. For example, for a tensor with shape (110,
///   120, 130), the 1st dimension is indexable from outer programs, then a unit operation handles
///   120x130 = 15,600 data. Then we call this function with ``unit_flop = 15600``. This value will
///   be set to 1 if too small.
/// - `mem_avail`: Memory available in MB. By default, it will check available memory in os system.
/// - `mem_factor`: factor for mem_avail, to avoid all memory consumed; should be smaller than 1,
///   recommended 0.7.
/// - `pre_flop`: Number of data preserved in memory. Unit in number.
pub fn calc_batch_size<T>(
    unit_flop: usize,
    mem_avail: Option<f64>,
    mem_factor: Option<f64>,
    pre_flop: Option<usize>,
) -> usize {
    let nbytes_dtype = std::mem::size_of::<T>();
    let unit_flop = unit_flop.max(1);
    let unit_mb = (unit_flop * nbytes_dtype) as f64 / 1024.0 / 1024.0;
    let pre_mb = pre_flop.unwrap_or(0) as f64 * nbytes_dtype as f64 / 1024.0 / 1024.0;
    let mem_factor = mem_factor.unwrap_or(0.7);
    let mem_avail_mb = mem_avail.unwrap_or_else(detect_available_memory_mb);
    let mem_avail_factored_mb = mem_avail.unwrap_or_else(detect_available_memory_mb) * mem_factor;
    let max_mb = mem_avail_factored_mb - pre_mb;

    if unit_mb > max_mb {
        println!("[WARN] Memory overflow when preparing batch number.");
        println!("       Current memory available {mem_avail_mb:10.3} MB, after allocation {max_mb:10.3} MB, minimum required per batch {unit_mb:10.3} MB");
    }
    let batch_size = (max_mb / unit_mb).floor().max(1.0).to_usize().unwrap();
    return batch_size;
}

/// Balance partition of indices.
///
/// This function is used to balance partition of indices, so that each partition has similar size.
/// This function mostly applied in shell-to-basis partition splitting.
///
/// # Parameters
///
/// - `indices`: List of indices to be partitioned. We assume this array is sorted and no elements
///   are the same value.
/// - `batch_size`: Maximum size of each partition.
///
/// # Example
///
/// ```rust
/// # use pyrest::utilities::memory_batch::blocksize_partition;
/// let indices = [1, 3, 6, 7, 10, 15, 16, 19];
/// let partitions = blocksize_partition(&indices, 4);
/// // A info of `[WARN] Batch size is too small: 15 - 10 > 4` will be printed.
/// assert_eq!(partitions, [[0, 1], [1, 3], [3, 4], [4, 5], [5, 7]]);
/// ```
pub fn blocksize_partition(indices: &[usize], batch_size: usize) -> Vec<[usize; 2]> {
    if batch_size == 0 {
        panic!("Batch size should not be zero.");
    }
    // handle special case
    if indices.len() <= 1 {
        return vec![];
    }

    let mut partitions = vec![0];
    let mut p0 = 0;
    let n = indices.len() - 1;
    for idx in 1..n {
        if indices[idx + 1] - indices[p0] > batch_size {
            if indices[idx] - indices[p0] > batch_size {
                println!("[WARN] Batch size is too small: {} - {} > {}", indices[idx], indices[p0], batch_size);
            }
            partitions.push(idx);
            p0 = idx;
        }
    }
    partitions.push(n);

    assert!(partitions.len() >= 2);
    let mut result = vec![];
    for i in 0..partitions.len() - 1 {
        result.push([partitions[i], partitions[i + 1]]);
    }
    return result;
}

/// Struct for memory estimation.
#[derive(Debug, Clone, Default)]
pub struct MemEstimate {
    /// Batched memory in DRAM, in number of elements.
    ///
    /// API caller must fill this field to get total memory estimation.
    pub batched: usize,

    /// Fixed memory in DRAM, in number of elements.
    ///
    /// This can be zero if no fixed memory consumption.
    pub fixed: usize,

    /// Fixed memory consumption in a thread, in number of elements.
    ///
    /// This can be zero if no fixed memory consumption.
    pub thread: usize,
}

impl MemEstimate {
    /// Print memory estimation info with data type.
    pub fn print_with_dtype<T>(&self) -> String {
        use std::fmt::Write;
        let nbytes_dtype = std::mem::size_of::<T>();
        let type_name = std::any::type_name::<T>();
        let batched_mb = (self.batched * nbytes_dtype) as f64 / 1024.0 / 1024.0;
        let fixed_mb = (self.fixed * nbytes_dtype) as f64 / 1024.0 / 1024.0;
        let thread_mb = (self.thread * nbytes_dtype) as f64 / 1024.0 / 1024.0;
        let mut output = String::new();
        writeln!(output, "[INFO] MemEstimate debug print:").unwrap();
        writeln!(output, "       dtype   = type {type_name} with {nbytes_dtype} bytes").unwrap();
        writeln!(output, "       batched = {batched_mb:10.3} MB").unwrap();
        writeln!(output, "       fixed   = {fixed_mb:10.3} MB").unwrap();
        writeln!(output, "       thread  = {thread_mb:10.3} MB").unwrap();
        // currently, we also print this to stdout
        print!("{output}");
        output
    }

    /// Estimate memory consumption in MB for given number of batches.
    pub fn estimate_mem<T>(&self, nbatch: usize) -> f64 {
        let nbytes_dtype = std::mem::size_of::<T>();
        let batched_mb = (self.batched * nbytes_dtype) as f64 / 1024.0 / 1024.0 * nbatch as f64;
        let fixed_mb = (self.fixed * nbytes_dtype) as f64 / 1024.0 / 1024.0;
        let thread_mb = (self.thread * nbytes_dtype) as f64 / 1024.0 / 1024.0 * rayon::current_num_threads() as f64;
        batched_mb + fixed_mb + thread_mb
    }
}

/// Calculate batch size within possible memory, by struct [`MemEstimate`].
///
/// For example, if we want to compute tensor (100, 100, 100), but only 50,000 memory available,
/// then this tensor should be splited into 20 batches.
///
/// ``flop`` in parameters is number of data, not refers to FLOPs.
///
/// This function requires generic `<T>`, which determines size of data.
///
/// # Parameters
///
/// - `mem_est`: Memory estimation struct.
/// - `mem_avail`: Memory available in MB. By default, it will check available memory in os system.
/// - `mem_factor`: factor for mem_avail, to avoid all memory consumed; should be smaller than 1,
///   recommended 0.7.
///
/// # Example
///
/// ```rust
/// # use pyrest::utilities::memory_batch::{MemEstimate, calc_batch_size_from_mem_estimate};
/// // make sure to let 6 threads in rayon
/// use rayon::prelude::*;
/// let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
/// let mem_est = MemEstimate {
///     batched: 200_000,
///     fixed: 500_000,
///     thread: 100_000,
/// };
/// let batch_size = pool.install(|| calc_batch_size_from_mem_estimate::<f64>(&mem_est, Some(500.0), Some(0.8), false));
/// println!("Calculated batch size: {}", batch_size);
/// assert_eq!(batch_size, 256);
/// ```
pub fn calc_batch_size_from_mem_estimate<T>(
    mem_est: &MemEstimate,
    mem_avail: Option<f64>,
    mem_factor: Option<f64>,
    print_warn: bool,
) -> usize {
    use rayon::prelude::*;
    let num_threads = rayon::current_num_threads();

    let nbytes_dtype = std::mem::size_of::<T>();
    let unit_flop = mem_est.batched.max(1);
    let unit_mb = (unit_flop * nbytes_dtype) as f64 / 1024.0 / 1024.0;
    let fixed_mb = mem_est.fixed as f64 * nbytes_dtype as f64 / 1024.0 / 1024.0;
    let thread_mb = mem_est.thread as f64 * nbytes_dtype as f64 / 1024.0 / 1024.0;
    let mem_factor = mem_factor.unwrap_or(0.7);
    let mem_avail_mb = mem_avail.unwrap_or_else(detect_available_memory_mb);
    let mem_avail_factored_mb = mem_avail.unwrap_or_else(detect_available_memory_mb) * mem_factor;
    let max_mb = mem_avail_factored_mb - fixed_mb - thread_mb * num_threads as f64;

    if unit_mb > max_mb && print_warn {
        eprintln!("[WARN] Memory overflow when preparing batch number.");
        eprintln!("       Current memory available {mem_avail_mb:10.3} MB, after allocation {max_mb:10.3} MB, minimum required per batch {unit_mb:10.3} MB");
        eprintln!("       Following debug info from MemEstimate:");
        mem_est.print_with_dtype::<T>();
    }
    let batch_size = (max_mb / unit_mb).floor().max(1.0).to_usize().unwrap();
    return batch_size;
}

/// Handle memory exceed situation (**this func returns `Result`, unwrap to panic**).
pub fn handle_memory_exceed(mem_to_use: f64, mem_avail: Option<f64>, abort_on_mem_exceed: bool) {
    let mem_avail_mb = mem_avail.unwrap_or_else(detect_available_memory_mb);
    if mem_to_use <= mem_avail_mb {
        return;
    }
    let msg = format!(
        "Memory usage exceeded: trying to use {mem_to_use:10.3} MB, but only {mem_avail_mb:10.3} MB available.\nIf you want to continue anyway, please set `abort_on_mem_exceed = false` in [ctrl] block of ctrl.in, but use that with caution and risk!"
    );
    if abort_on_mem_exceed {
        panic!("[ERROR] {msg}")
    } else {
        eprintln!("[WARN] {msg}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_batch_size_from_mem_estimate() {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        let mem_est = MemEstimate { batched: 200_000, fixed: 500_000, thread: 100_000 };
        let batch_size =
            pool.install(|| calc_batch_size_from_mem_estimate::<f64>(&mem_est, Some(500.0), Some(0.8), true));
        println!("Calculated batch size: {batch_size}");
        assert_eq!(batch_size, 256);
    }
}
