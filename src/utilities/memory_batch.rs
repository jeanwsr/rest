//! Memory size detection and batching utilities.

use num_traits::ToPrimitive;

/// Detect available memory in system in MB.
pub fn detect_available_memory_mb() -> f64 {
    let sys = sysinfo::System::new_all();
    (sys.total_memory() - sys.used_memory()) as f64 / 1024.0 / 1024.0
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
            let sys = sysinfo::System::new_all();
            sys.used_memory() as f64 / 1024.0 / 1024.0
        },
        "proc" => {
            let sys = sysinfo::System::new_all();
            let pid = sysinfo::get_current_pid().unwrap();
            let process = sys.process(pid).unwrap();
            process.memory() as f64 / 1024.0 / 1024.0
        },
        _ => {
            panic!("Unknown use_case for detect_used_memory_mb: {}", use_case);
        },
    }
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
/// # use pyrest::grad::rhf::blocksize_partition;
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
/// let batch_size = pool.install(|| calc_batch_size_from_mem_estimate::<f64>(&mem_est, Some(500.0), Some(0.8)));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_batch_size_from_mem_estimate() {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        let mem_est = MemEstimate { batched: 200_000, fixed: 500_000, thread: 100_000 };
        let batch_size = pool.install(|| calc_batch_size_from_mem_estimate::<f64>(&mem_est, Some(500.0), Some(0.8), true));
        println!("Calculated batch size: {batch_size}");
        assert_eq!(batch_size, 256);
    }
}
