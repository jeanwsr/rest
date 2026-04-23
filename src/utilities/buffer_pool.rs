#![warn(unused)]
use std::sync::{Mutex, Arc};

/// A thread-safe buffer pool for reusing buffers of type T.
pub struct BufferPool<'a, T> {
    pub pool: Arc<Mutex<Vec<T>>>,
    // the `'a` here can avoid the 'static lifetime requirement for the init function, allowing it to capture
    // non-static references if needed (and lives longer than the BufferPool itself)
    pub init: Box<dyn Fn() -> T + Send + Sync + 'a>,
}

impl<'a, T> BufferPool<'a, T> {
    /// Creates a new BufferPool with the given initialization function.
    pub fn new(init: impl Fn() -> T + Send + Sync + 'a) -> Self {
        BufferPool { pool: Arc::new(Mutex::new(Vec::new())), init: Box::new(init) }
    }

    pub fn get(&self) -> T {
        if let Ok(mut guard) = self.pool.lock() {
            if let Some(buffer) = guard.pop() {
                return buffer;
            }
        }
        (self.init)()
    }

    pub fn put(&self, buffer: T) {
        if let Ok(mut guard) = self.pool.lock() {
            guard.push(buffer);
        }
    }
}
