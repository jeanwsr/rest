use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::SeqCst);
}

/// Append `name` to `./matvec_count.txt` when tracing is enabled.
/// Opens and closes the file each call (fine for profiling, not performance).
pub fn trace(name: &str) {
    if ENABLED.load(Ordering::SeqCst) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true).create(true)
            .open("./matvec_count.txt")
        {
            let _ = writeln!(f, "{}", name);
        }
    }
}
