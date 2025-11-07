//! Memory size detection and batching utilities.

use num_traits::ToPrimitive;

/// Detect available memory in system in MB.
pub fn detect_available_memory_mb() -> f64 {
    let sys = sysinfo::System::new_all();
    (sys.total_memory() - sys.used_memory()) as f64 / 1024.0 / 1024.0
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
    let mem_avail_mb = mem_avail.unwrap_or_else(detect_available_memory_mb) * mem_factor;
    let max_mb = mem_avail_mb - pre_mb;

    if unit_mb > max_mb {
        println!("[Warn] Memory overflow when preparing batch number.");
        println!("Current memory available {:10.3} MB, minimum required {:10.3} MB", max_mb, unit_mb);
    }
    let batch_size = (max_mb / unit_mb).max(1.0).to_usize().unwrap();
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
/// // A info of `[Warn] Batch size is too small: 15 - 10 > 4` will be printed.
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
                println!("[Warn] Batch size is too small: {} - {} > {}", indices[idx], indices[p0], batch_size);
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
