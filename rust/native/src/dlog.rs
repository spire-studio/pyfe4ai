//! Shared pieces of the bounded discrete logarithms (ristretto255 in `ec`, GT
//! in `pairing`): the bound cap, the baby-step table size, and a small LRU
//! cache of tables capped in count and bytes, so that tables are rebuilt
//! rather than kept for the life of the process.

use std::sync::{Arc, Mutex};

/// Largest bound of a discrete logarithm (|x| <= 2^48): above it the search
/// takes hours, and `2 bound + 1` must not wrap. Larger bounds raise ValueError.
pub const MAX_BOUND: u64 = 1 << 48;
/// Baby steps per square root of the range (fewer giant steps for a table built once).
pub const BABY_FACTOR: u64 = 16;
/// Most baby steps of one table (about 100 MB): beyond it, more giant steps.
pub const MAX_BABY: u64 = 1 << 21;
/// Bytes of one table entry (a 32-byte key, the index, the hash map's overhead).
pub const ENTRY_BYTES: usize = 48;

/// Baby steps of a table for |x| <= bound: about BABY_FACTOR sqrt(2 bound + 1)
/// for a cached table, sqrt(2 bound + 1) for a one-shot search (`cached` false).
pub fn baby_steps(bound: u64, cached: bool) -> u64 {
    let span = 2 * bound + 1; // bound <= MAX_BOUND: no wrap
    let factor = if cached { BABY_FACTOR } else { 1 };
    span.min(((span as f64).sqrt() as u64 + 1) * factor).min(MAX_BABY)
}

/// A process-wide cache of tables, least recently used evicted first.
pub struct TableCache<K: PartialEq, T> {
    max_tables: usize,
    max_bytes: usize,
    entries: Mutex<Vec<(K, Arc<T>, usize)>>, // least recently used first
}

impl<K: PartialEq, T> TableCache<K, T> {
    pub const fn new(max_tables: usize, max_bytes: usize) -> Self {
        TableCache { max_tables, max_bytes, entries: Mutex::new(Vec::new()) }
    }

    /// (tables, estimated bytes) held now.
    pub fn stats(&self) -> (usize, usize) {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        (entries.len(), entries.iter().map(|e| e.2).sum())
    }

    /// The table of `key`, built by `build` (without the cache's lock) if absent;
    /// `bytes` estimates its size. A table larger than the whole cache is not kept.
    pub fn get_or_build(&self, key: K, bytes: usize, build: impl FnOnce() -> T) -> Arc<T> {
        let lock = || self.entries.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut entries = lock();
            if let Some(i) = entries.iter().position(|(k, _, _)| *k == key) {
                let e = entries.remove(i);
                let t = e.1.clone();
                entries.push(e);
                return t;
            }
        }
        let t = Arc::new(build());
        if bytes <= self.max_bytes {
            let mut entries = lock();
            if let Some((_, other, _)) = entries.iter().find(|(k, _, _)| *k == key) {
                return other.clone(); // built meanwhile by another thread
            }
            entries.push((key, t.clone(), bytes));
            let mut total: usize = entries.iter().map(|e| e.2).sum();
            while entries.len() > self.max_tables || total > self.max_bytes {
                total -= entries.remove(0).2;
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_evicts_least_recently_used() {
        let c: TableCache<u64, u64> = TableCache::new(2, 100);
        c.get_or_build(1, 10, || 1);
        c.get_or_build(2, 10, || 2);
        c.get_or_build(1, 10, || 99); // hit: 1 is now the most recent
        c.get_or_build(3, 10, || 3); // evicts 2
        assert_eq!(*c.get_or_build(1, 10, || 99), 1);
        assert_eq!(*c.get_or_build(2, 10, || 22), 22);
        assert_eq!(*c.get_or_build(4, 1000, || 4), 4); // too large: not kept
        assert_eq!(c.stats(), (2, 20));
    }

    #[test]
    fn baby_steps_are_capped() {
        assert_eq!(baby_steps(0, true), 1);
        assert_eq!(baby_steps(100, false), 15);
        assert_eq!(baby_steps(MAX_BOUND, true), MAX_BABY);
    }
}
