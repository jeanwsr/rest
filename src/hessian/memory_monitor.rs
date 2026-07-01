//! Lightweight process-memory monitoring utilities for the analytical Hessian.
//!
//! Provides:
//!   * [`current_rss_mb`] — read current resident set size (RSS) of this process.
//!   * [`print_system_size`] — dump all system-size parameters relevant to memory analysis.
//!   * [`MemMonitor`] — a background thread that polls RSS, tracks the peak, and
//!     aborts the process if RSS exceeds a user-configured GiB limit.
//!
//! Only Linux is supported (reads `/proc/self/status`). On other platforms the
//! RSS reader returns 0.0 and the monitor degrades to a no-op peak tracker.

use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Return freed arena memory to the OS by calling glibc's `malloc_trim`.
///
/// Call this between Hessian stages so that the RSS baseline going into the
/// next stage is as low as possible. glibc's default allocator keeps freed
/// heap pages in its arena rather than `munmap`-ing them, which inflates the
/// apparent peak of the following stage. `malloc_trim(0)` releases everything
/// it safely can. No-op (returns 0) on non-glibc allocators.
pub fn trim_to_os(print_level: usize) {
    let rss_before = current_rss_mb();
    unsafe { let _ = libc::malloc_trim(0); }
    let rss_after = current_rss_mb();
    if print_level > 1 {
        println!(
            "  [mem] malloc_trim: RSS {:.3} -> {:.3} MiB (released {:.3} MiB to OS)",
            rss_before, rss_after, (rss_before - rss_after).max(0.0),
        );
    }
}

/// Read current RSS of this process in MiB by parsing `/proc/self/status`.
/// Returns 0.0 on non-Linux platforms or on read error.
pub fn current_rss_mb() -> f64 {
    let txt = match fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0.0,
    };
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:\t      1234 kB"
            let mut iter = rest.split_whitespace();
            let val: f64 = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let unit = iter.next().unwrap_or("kB");
            return match unit {
                "kB" => val / 1024.0,
                "MB" => val,
                "GB" => val * 1024.0,
                _ => val / 1024.0,
            };
        }
    }
    0.0
}

/// Handle to a background memory-monitoring thread.
///
/// The thread polls the process RSS every `poll_interval` and:
///   * updates an atomic peak (in MiB × 1024, i.e. KiB, to keep integer math),
///   * aborts the process via `std::process::abort()` if RSS exceeds `limit_gb`.
///
/// Call [`MemMonitor::stage_peak_mb`] to read (and reset) the peak for the
/// current stage, and [`MemMonitor::stop`] when the Hessian is done.
pub struct MemMonitor {
    /// Atomic peak RSS in KiB since last reset.
    peak_kib: Arc<AtomicU64>,
    /// Atomic running flag — set false by `stop()` to terminate the thread.
    running: Arc<AtomicBool>,
    /// Limit in GiB; `None` disables the abort check.
    limit_gb: Option<f64>,
    handle: Option<JoinHandle<()>>,
}

impl MemMonitor {
    /// Spawn a monitor thread.
    ///
    /// `poll_interval` controls how often RSS is sampled. `limit_gb` of `None`
    /// means "track peak only, never abort".
    pub fn start(limit_gb: Option<f64>, poll_interval: Duration) -> Self {
        let peak_kib = Arc::new(AtomicU64::new(0));
        let running = Arc::new(AtomicBool::new(true));
        let peak_clone = Arc::clone(&peak_kib);
        let running_clone = Arc::clone(&running);
        let limit_mib = limit_gb.map(|g| g * 1024.0);

        let handle = thread::Builder::new()
            .name("rest_mem_monitor".to_string())
            .spawn(move || {
                while running_clone.load(Ordering::Relaxed) {
                    let rss = current_rss_mb();
                    let kib = (rss * 1024.0) as u64;
                    // Update peak
                    let mut cur = peak_clone.load(Ordering::Relaxed);
                    while kib > cur {
                        match peak_clone.compare_exchange_weak(
                            cur, kib,
                            Ordering::Relaxed, Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(v) => cur = v,
                        }
                    }
                    // Check limit
                    if let Some(lim_mib) = limit_mib {
                        if rss > lim_mib {
                            eprintln!(
                                "\n[FATAL] REST process RSS ({:.3} MiB = {:.3} GiB) exceeded \
                                 max_memory_gb limit ({:.3} GiB). Aborting.",
                                rss, rss / 1024.0, lim_mib / 1024.0
                            );
                            std::process::abort();
                        }
                    }
                    thread::sleep(poll_interval);
                }
            })
            .ok();

        MemMonitor {
            peak_kib,
            running,
            limit_gb,
            handle,
        }
    }

    /// Start a monitor with no GB limit (peak tracking only), 20 ms poll.
    pub fn no_limit() -> Self {
        Self::start(None, Duration::from_millis(20))
    }

    /// Return the current limit in GiB, if any.
    pub fn limit_gb(&self) -> Option<f64> { self.limit_gb }

    /// Read and reset the peak RSS (in MiB) accumulated since the last call.
    pub fn stage_peak_mb(&self) -> f64 {
        let kib = self.peak_kib.swap(0, Ordering::Relaxed);
        (kib as f64) / 1024.0
    }

    /// Read the peak without resetting it.
    pub fn peek_peak_mb(&self) -> f64 {
        (self.peak_kib.load(Ordering::Relaxed) as f64) / 1024.0
    }

    /// Stop the monitor thread and join it.
    pub fn stop(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Print all system-size parameters that influence the Hessian memory footprint.
///
/// `ngrids` is the number of DFT integration grid points (0 for pure HF / when
/// grids are not available). `naux` is the auxiliary-basis size.
pub fn print_system_size(
    label: &str,
    natm: usize,
    nao: usize,
    nocc: usize,
    naux: usize,
    ngrids: usize,
) {
    let nvirt = nao.saturating_sub(nocc);
    let n3 = natm * 3;
    let rss = current_rss_mb();
    println!("\n=== Hessian system-size report ({}) ===", label);
    println!("  atoms (natm)                  : {}", natm);
    println!("  AO basis functions (nao)      : {}", nao);
    println!("  occupied orbitals (nocc)      : {}", nocc);
    println!("  virtual orbitals (nvirt)      : {}", nvirt);
    println!("  aux basis functions (naux)    : {}", naux);
    println!("  DFT grid points (ngrids)      : {}", ngrids);
    println!("  Hessian matrix dimension (n3) : {} x {}", n3, n3);
    println!("  ---------------------------------------------");
    // Rough footprint estimates (f64, bytes = 8):
    let kb = 1024.0;
    let mb = 1024.0 * kb;
    let gb = 1024.0 * mb;
    let f64b = 8.0;
    println!("  Estimated footprint of key arrays (f64):");
    println!("    dm0 [nao,nao]               : {:.3} MiB", (nao * nao) as f64 * f64b / mb);
    println!("    int3c2e [nao,nao,naux]      : {:.3} GiB", (nao * nao * naux) as f64 * f64b / gb);
    println!("    int2c2e [naux,naux]         : {:.3} MiB", (naux * naux) as f64 * f64b / mb);
    println!("    h_partial [n3,n3]           : {:.3} MiB", (n3 * n3) as f64 * f64b / mb);
    println!("    e1 [n3,n3]                  : {:.3} MiB", (n3 * n3) as f64 * f64b / mb);
    if ngrids > 0 {
        // ao deriv-4 block: roughly [nderiv, nao, ngrids_blk]
        // Per grid point, deriv-4 has 35 components for GGA, 15 for LDA. Use 35 as upper bound.
        let ao_full = (nao * ngrids * 35) as f64 * f64b;
        println!("    ao(deriv4) [nao,ngrids,35]  : {:.3} GiB", ao_full / gb);
        println!("    rho on grid (per block)     : negligible vs ao");
    }
    println!("  ---------------------------------------------");
    println!("  Current process RSS at start  : {:.3} MiB ({:.3} GiB)", rss, rss / 1024.0);
}
