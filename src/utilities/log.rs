use log::{LevelFilter};

pub fn printlevel2loglevel(level: usize) -> LevelFilter {
    let l = match level {
        0 => LevelFilter::Info,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        3.. => LevelFilter::Trace,
        _ => LevelFilter::Info,
    };
    l
}
