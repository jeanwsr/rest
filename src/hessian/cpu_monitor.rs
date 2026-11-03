//! Lightweight CPU / thread utilization monitor for the analytical Hessian.
//!
//! Complements [`super::memory_monitor`] by answering the question
//! *"how many cores did this stage actually use?"*. This is essential for
//! diagnosing parallel-efficiency regressions in `calc_ej_ek` and friends,
//! where the outer atom loops are sequential and only BLAS parallelises
//! internally — for small molecules BLAS may use only 1–2 threads even
//! when the user requested 8.
//!
//! # What it measures
//! * **Process CPU jiffies** (`utime + stime` from `/proc/self/stat`) —
//!   the total CPU·s consumed by the process across all cores.
//! * **System-wide busy jiffies** (from `/proc/stat`) — sum of all non-idle
//!   time across all logical CPUs, used to detect contention from other
//!   processes.
//! * **Instantaneous process utilization** sampled every `poll_interval` —
//!   `Δcpu_jiffies / (Δwall · num_cpus · ticks_per_sec) ∈ [0, num_cpus]`.
//! * **Average utilization per labelled section** via [`CpuSection`] guards.
//! * **Per-section peak RSS delta** by sampling [`current_rss_mb`] inside the
//!   background thread.
//!
//! # Threading model
//! The monitor owns one background thread named `rest_cpu_monitor` that only
//! reads `/proc` files and pushes samples into a `Mutex<Vec<..>>`. Section
//! guards are `Send` and may be created from any worker thread; the only
//! state they touch on drop is the shared `Arc<Mutex<Vec<SectionRecord>>>`.
//!
//! Only Linux is supported. On other platforms readers return 0 and the
//! monitor degrades to a wall-clock-only peak tracker (still useful for
//! timing breakdowns).
//!
//! [`current_rss_mb`]: super::memory_monitor::current_rss_mb

use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::memory_monitor::current_rss_mb;

// ─────────────────────────────────────────────────────────────────────
// /proc readers
// ─────────────────────────────────────────────────────────────────────

/// Read process CPU jiffies (utime + stime) from `/proc/self/stat`.
///
/// Returns 0 on non-Linux platforms or on read/parse error. The values are
/// in clock-tick units; convert to seconds with [`ticks_per_second`].
pub fn process_cpu_jiffies() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let txt = match fs::read_to_string("/proc/self/stat") {
            Ok(s) => s,
            Err(_) => return 0,
        };
        // Field 14 is utime, field 15 is stime (1-based). The comm field (2)
        // is wrapped in parens and may contain whitespace, so skip past the
        // closing ')' before splitting on whitespace.
        let after_comm = match txt.rfind(')') {
            Some(i) => &txt[i + 1..],
            None => return 0,
        };
        let mut fields = after_comm.split_whitespace();
        // Skip fields 3..=13 (state .. signalblock) to reach field 14.
        // fields iterator is now positioned at field 3.
        for _ in 3..=13 {
            fields.next();
        }
        let utime: u64 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let stime: u64 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        utime.saturating_add(stime)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Read system-wide CPU jiffies from `/proc/stat`. Returns `(busy, total)`
/// where `busy` excludes idle/iowait and `total` includes everything.
///
/// Both are 0 on non-Linux or on parse failure.
pub fn system_cpu_jiffies() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let txt = match fs::read_to_string("/proc/stat") {
            Ok(s) => s,
            Err(_) => return (0, 0),
        };
        let first = match txt.lines().next() {
            Some(l) => l,
            None => return (0, 0),
        };
        // Format: "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
        let nums: Vec<u64> = first
            .split_whitespace()
            .skip(1)
            .filter_map(|s| s.parse::<u64>().ok())
            .collect();
        if nums.len() < 4 {
            return (0, 0);
        }
        let user = nums[0];
        let nice = nums[1];
        let system = nums[2];
        let idle = nums[3];
        let iowait = *nums.get(4).unwrap_or(&0);
        let irq = *nums.get(5).unwrap_or(&0);
        let softirq = *nums.get(6).unwrap_or(&0);
        let steal = *nums.get(7).unwrap_or(&0);
        let busy = user + nice + system + irq + softirq + steal;
        let total = busy + idle + iowait;
        (busy, total)
    }
    #[cfg(not(target_os = "linux"))]
    {
        (0, 0)
    }
}

/// Number of clock ticks per second (`USER_HZ` / `CLOCKS_PER_SEC` on Linux).
/// Falls back to `sysconf(_SC_CLK_TCK)` via libc when available, else 100.
pub fn ticks_per_second() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // _SC_CLK_TCK = 2 in unistd.h
        let v = unsafe { libc::sysconf(2) };
        if v > 0 {
            return v as u64;
        }
    }
    100
}

/// Count logical CPUs by counting `processor` entries in `/proc/cpuinfo`.
/// Falls back to `1` on failure. Used to convert "CPU·seconds consumed" into
/// "average cores busy" (utilization as a fraction in [0, num_cpus]).
pub fn num_logical_cpus() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = fs::read_to_string("/proc/cpuinfo") {
            let n = txt
                .lines()
                .filter(|l| l.starts_with("processor"))
                .count();
            if n > 0 {
                return n;
            }
        }
    }
    // Last-resort fallbacks via env / rayon.
    std::env::var("OMP_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| rayon::current_num_threads())
}

// ─────────────────────────────────────────────────────────────────────
// Samples and section records
// ─────────────────────────────────────────────────────────────────────

/// A single point-in-time snapshot of process / system CPU state.
#[derive(Clone, Copy, Default)]
pub struct CpuSample {
    /// Wall-clock offset from monitor start (microseconds).
    pub wall_us: u64,
    /// Process `utime + stime` in jiffies.
    pub proc_jiffies: u64,
    /// System-wide busy jiffies (non-idle, all cores).
    pub sys_busy_jiffies: u64,
    /// System-wide total jiffies (including idle).
    pub sys_total_jiffies: u64,
    /// RSS in MiB.
    pub rss_mb: f64,
}

/// Aggregate statistics for one labelled section.
pub struct SectionRecord {
    pub label: String,
    pub wall_secs: f64,
    /// Average cores actively used by the REST process during this section.
    /// Range: `[0.0, num_logical_cpus()]`. `1.0` means single-threaded.
    pub avg_cores: f64,
    /// Peak instantaneous cores observed by the background sampler during
    /// the section (0.0 if no samples fell inside the window).
    pub peak_cores: f64,
    /// Minimum instantaneous cores (excluding the very first sample).
    pub min_cores: f64,
    /// Number of background samples that fell inside the section window.
    pub n_samples: usize,
    /// System-wide average cores busy (includes other processes). Useful to
    /// detect "we used 2 cores but something else was stealing CPU".
    pub sys_avg_cores_busy: f64,
    /// RSS peak (MiB) reached during the section.
    pub rss_peak_mb: f64,
    /// RSS at section start (MiB), for delta reporting.
    pub rss_start_mb: f64,
}

/// Background CPU sampler. Cheap to start/stop; safe to share `&Arc<Self>`
/// across threads.
pub struct CpuMonitor {
    inner: Arc<MonitorInner>,
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

struct MonitorInner {
    start_wall: Instant,
    samples: Mutex<Vec<CpuSample>>,
    sections: Mutex<Vec<SectionRecord>>,
    tick_hz: u64,
    num_cpus: usize,
}

impl CpuMonitor {
    /// Spawn a background sampler polling every `poll_interval`.
    pub fn start(poll_interval: Duration) -> Self {
        let inner = Arc::new(MonitorInner {
            start_wall: Instant::now(),
            samples: Mutex::new(Vec::with_capacity(1024)),
            sections: Mutex::new(Vec::new()),
            tick_hz: ticks_per_second(),
            num_cpus: num_logical_cpus(),
        });
        let running = Arc::new(AtomicBool::new(true));
        let inner_clone = Arc::clone(&inner);
        let running_clone = Arc::clone(&running);

        let handle = thread::Builder::new()
            .name("rest_cpu_monitor".to_string())
            .spawn(move || {
                // Seed sample[0] so the first delta in the loop is well-defined.
                push_sample(&inner_clone, 0);
                while running_clone.load(Ordering::Relaxed) {
                    thread::sleep(poll_interval);
                    let wall_us = inner_clone.start_wall.elapsed().as_micros() as u64;
                    push_sample(&inner_clone, wall_us);
                }
            })
            .ok();

        CpuMonitor {
            inner,
            running,
            handle,
        }
    }

    /// Convenience constructor with a 25 ms poll — fine-grained enough to
    /// catch utilisation spikes inside G-terms that take as little as a
    /// few hundred microseconds on H2O-sized systems.
    pub fn default_period() -> Self {
        Self::start(Duration::from_millis(25))
    }

    /// Number of logical CPUs detected at construction time.
    pub fn num_cpus(&self) -> usize { self.inner.num_cpus }

    /// Clock-tick frequency (typically 100 Hz on Linux).
    pub fn tick_hz(&self) -> u64 { self.inner.tick_hz }

    /// Wall-clock seconds since the monitor started.
    pub fn elapsed_secs(&self) -> f64 {
        self.inner.start_wall.elapsed().as_secs_f64()
    }

    /// Begin a labelled section. Returns a guard that finalises the record
    /// on drop, so `let _s = mon.section("g4");` instruments any scope.
    pub fn section(&self, label: impl Into<String>) -> CpuSection<'_> {
        let label = label.into();
        let start_wall_us = self.inner.start_wall.elapsed().as_micros() as u64;
        let start_cpu = process_cpu_jiffies();
        let rss_start = current_rss_mb();
        CpuSection {
            monitor: self,
            label,
            start_wall_us,
            start_cpu,
            rss_start,
        }
    }

    /// Record a completed section (called by [`CpuSection::drop`]).
    fn record_section(&self, rec: SectionRecord) {
        self.inner.sections.lock().unwrap().push(rec);
    }

    /// Snapshot of the current sample buffer (for callers that want to do
    /// their own analysis). Returns a clone; cheap if there are few samples.
    pub fn samples_snapshot(&self) -> Vec<CpuSample> {
        self.inner.samples.lock().unwrap().clone()
    }

    /// Stop the background sampler and join it. Optional — [`Drop`] also
    /// stops the thread, so callers can simply let the monitor go out of
    /// scope.
    pub fn stop(mut self) {
        self.signal_stop();
    }

    /// Signal the background thread to terminate and join it. Called by
    /// [`Drop`] automatically; exposed for callers that want to flush the
    /// sample buffer before reading [`CpuMonitor::samples_snapshot`].
    pub fn signal_stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    /// Print a per-section utilisation table plus totals.
    pub fn report(&self) {
        let sections = self.inner.sections.lock().unwrap();
        let tick_hz = self.inner.tick_hz.max(1) as f64;
        let ncpu = self.inner.num_cpus;
        let total_wall: f64 = sections.iter().map(|s| s.wall_secs).sum();
        let total_cpu_secs: f64 = sections
            .iter()
            .map(|s| s.avg_cores * s.wall_secs)
            .sum();
        let overall_cores = if total_wall > 0.0 {
            total_cpu_secs / total_wall
        } else {
            0.0
        };

        println!();
        println!("=== CPU utilisation report ===");
        println!("  logical CPUs detected     : {}", ncpu);
        println!("  clock tick frequency      : {:.0} Hz", tick_hz);
        println!("  background poll samples   : {}", self.inner.samples.lock().unwrap().len());
        println!("  {:<22} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "section", "wall(s)", "avg cores", "peak", "min", "RSS peak");
        println!("  {:-<82}", "");
        for s in sections.iter() {
            println!(
                "  {:<22} {:>10.4} {:>10.3} {:>10.3} {:>10.3} {:>9.1} MiB",
                s.label, s.wall_secs, s.avg_cores, s.peak_cores, s.min_cores, s.rss_peak_mb,
            );
        }
        println!("  {:-<82}", "");
        println!(
            "  {:<22} {:>10.4} {:>10.3}  (overall = {:.1}% of {} cores)",
            "TOTAL", total_wall, overall_cores, overall_cores / ncpu.max(1) as f64 * 100.0, ncpu,
        );
        println!();
    }

    /// Compute per-section aggregates from the sample window at finalisation
    /// time. Called by [`CpuSection::drop`].
    fn finalize_section(
        &self,
        label: String,
        start_wall_us: u64,
        start_cpu: u64,
        rss_start: f64,
    ) {
        let end_wall_us = self.inner.start_wall.elapsed().as_micros() as u64;
        let end_cpu = process_cpu_jiffies();
        let wall_secs = (end_wall_us.saturating_sub(start_wall_us)) as f64 / 1e6;
        let cpu_secs = (end_cpu.saturating_sub(start_cpu)) as f64 / self.inner.tick_hz.max(1) as f64;
        let avg_cores = if wall_secs > 0.0 { cpu_secs / wall_secs } else { 0.0 };

        // Scan background samples that fell inside [start_wall_us, end_wall_us]
        // for instantaneous min / peak cores and RSS peak.
        let samples = self.inner.samples.lock().unwrap();
        let mut peak_cores = 0.0f64;
        let mut min_cores = f64::INFINITY;
        let mut n_samples = 0usize;
        let mut rss_peak = rss_start;
        let mut prev: Option<CpuSample> = None;
        let mut sys_busy_start: Option<u64> = None;
        let mut sys_busy_end: u64 = 0;
        for s in samples.iter() {
            if s.wall_us < start_wall_us {
                prev = Some(*s);
                continue;
            }
            if s.wall_us > end_wall_us {
                break;
            }
            // First sample inside the window — anchor system-busy counter.
            if sys_busy_start.is_none() {
                sys_busy_start = Some(s.sys_busy_jiffies);
            }
            sys_busy_end = s.sys_busy_jiffies;
            if s.rss_mb > rss_peak {
                rss_peak = s.rss_mb;
            }
            if let Some(p) = prev {
                let dt_us = s.wall_us.saturating_sub(p.wall_us) as f64;
                let dj = s.proc_jiffies.saturating_sub(p.proc_jiffies) as f64;
                if dt_us > 0.0 {
                    let inst_secs = dj / self.inner.tick_hz.max(1) as f64;
                    let cores = inst_secs / (dt_us / 1e6);
                    if cores > peak_cores { peak_cores = cores; }
                    if cores < min_cores { min_cores = cores; }
                    n_samples += 1;
                }
            }
            prev = Some(*s);
        }
        if !min_cores.is_finite() { min_cores = 0.0; }
        let sys_avg_cores_busy = if let Some(s0) = sys_busy_start {
            let dt_wall = end_wall_us.saturating_sub(start_wall_us) as f64 / 1e6;
            if dt_wall > 0.0 {
                (sys_busy_end.saturating_sub(s0)) as f64
                    / self.inner.tick_hz.max(1) as f64
                    / dt_wall
            } else { 0.0 }
        } else { 0.0 };

        self.record_section(SectionRecord {
            label,
            wall_secs,
            avg_cores,
            peak_cores,
            min_cores,
            n_samples,
            sys_avg_cores_busy,
            rss_peak_mb: rss_peak,
            rss_start_mb: rss_start,
        });
    }
}

fn push_sample(inner: &Arc<MonitorInner>, wall_us: u64) {
    let (sys_busy, sys_total) = system_cpu_jiffies();
    let s = CpuSample {
        wall_us,
        proc_jiffies: process_cpu_jiffies(),
        sys_busy_jiffies: sys_busy,
        sys_total_jiffies: sys_total,
        rss_mb: current_rss_mb(),
    };
    let mut buf = inner.samples.lock().unwrap();
    // Cap memory: keep at most 200_000 samples (~12 MiB at 80 B/sample).
    if buf.len() < 200_000 {
        buf.push(s);
    }
}

/// Scope guard for a labelled section. Finalises the record on drop.
pub struct CpuSection<'a> {
    monitor: &'a CpuMonitor,
    label: String,
    start_wall_us: u64,
    start_cpu: u64,
    rss_start: f64,
}

impl<'a> Drop for CpuSection<'a> {
    fn drop(&mut self) {
        self.monitor.finalize_section(
            std::mem::take(&mut self.label),
            self.start_wall_us,
            self.start_cpu,
            self.rss_start,
        );
    }
}

impl Drop for CpuMonitor {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Thread / runtime configuration report
// ─────────────────────────────────────────────────────────────────────

/// Snapshot of every environment variable and runtime value that influences
/// how many threads BLAS / rayon will spawn. Printed once at the start of
/// `calc_ej_ek` so the user can see e.g. `OMP_NUM_THREADS=2` immediately.
#[derive(Default)]
pub struct ThreadReport {
    pub omp_num_threads_env: Option<String>,
    pub mkl_num_threads_env: Option<String>,
    pub openblas_num_threads_env: Option<String>,
    pub rayon_num_threads_env: Option<String>,
    pub omp_runtime_threads: usize,
    pub rayon_pool_threads: usize,
    pub num_logical_cpus: usize,
}

impl ThreadReport {
    /// Collect from environment + runtime introspection.
    pub fn collect() -> Self {
        ThreadReport {
            omp_num_threads_env: std::env::var("OMP_NUM_THREADS").ok(),
            mkl_num_threads_env: std::env::var("MKL_NUM_THREADS").ok(),
            openblas_num_threads_env: std::env::var("OPENBLAS_NUM_THREADS").ok(),
            rayon_num_threads_env: std::env::var("RAYON_NUM_THREADS").ok(),
            omp_runtime_threads: omp_runtime_max_threads(),
            rayon_pool_threads: rayon::current_num_threads(),
            num_logical_cpus: num_logical_cpus(),
        }
    }

    /// Pretty-print as a labelled block. `header` is typically a stage name
    /// like `"calc_ej_ek entry"`.
    pub fn print(&self, header: &str) {
        println!();
        println!("=== Thread configuration report ({}) ===", header);
        println!("  OMP_NUM_THREADS env       : {}",
            self.omp_num_threads_env.as_deref().unwrap_or("(unset)"));
        println!("  MKL_NUM_THREADS env       : {}",
            self.mkl_num_threads_env.as_deref().unwrap_or("(unset)"));
        println!("  OPENBLAS_NUM_THREADS env  : {}",
            self.openblas_num_threads_env.as_deref().unwrap_or("(unset)"));
        println!("  RAYON_NUM_THREADS env     : {}",
            self.rayon_num_threads_env.as_deref().unwrap_or("(unset)"));
        println!("  omp_get_max_threads()     : {}", self.omp_runtime_threads);
        println!("  rayon::current_num_threads: {}", self.rayon_pool_threads);
        println!("  logical CPUs detected     : {}", self.num_logical_cpus);
        // Flag the most common mis-configuration immediately.
        if self.omp_runtime_threads < self.num_logical_cpus {
            println!("  ⚠ BLAS will use at most {} threads (out of {} CPUs) —",
                self.omp_runtime_threads, self.num_logical_cpus);
            println!("    outer atom loops in calc_ej_ek are SEQUENTIAL, so this");
            println!("    caps the parallelism of the entire stage.");
        }
        if self.rayon_pool_threads < self.num_logical_cpus {
            println!("  ⚠ rayon pool has only {} threads — par_iter over atoms will",
                self.rayon_pool_threads);
            println!("    not saturate all {} CPUs even after parallelisation.",
                self.num_logical_cpus);
        }
        println!();
    }
}

/// Call `omp_get_max_threads()` via the same wrapper used elsewhere in REST.
/// Returns 1 on non-OpenMP builds.
fn omp_runtime_max_threads() -> usize {
    // `omp_get_num_threads_wrapper` actually returns `omp_get_max_threads()`
    // (see rest_tensors/src/matrix/matrix_blas_lapack.rs:32-40). We use the
    // same crate-internal wrapper to stay consistent with the rest of REST.
    tensors::matrix_blas_lapack::omp_get_num_threads_wrapper()
}
