//! Shared pieces of the bounded discrete logarithms (ristretto255 in `ec`, GT
//! in `pairing`): the bound cap, the baby-step table size, and a small LRU
//! cache of tables capped in count and bytes, so that tables are rebuilt
//! rather than kept for the life of the process.
//!
//! Sizes are those std's HashMap (hashbrown) allocates: a table of m entries
//! has the smallest power of two of buckets that holds m at a load of 7/8,
//! each the entry plus one control byte (between 1.15 and 2.3 times the
//! entries' own bytes). One table may take MAX_TABLE_BYTES, so it holds at
//! most 1,835,008 baby steps (2^21 buckets, 86 MB for ristretto255, 69 MB for
//! GT). The trade-off: above bounds of about 2^33 (where 16 sqrt(2 bound + 1)
//! exceeds that) a dlog takes about 2 bound / 1.8e6 giant steps instead of
//! sqrt(2 bound + 1) / 16. ristretto255 at 10^11 (M2 Max): the 86 MB table takes
//! 7 s to build and a dlog 0.25 s, where the uncapped table of 7.2e6 steps
//! (344 MB) took 28 s and 50 ms. A larger MAX_TABLE_BYTES (and CACHE_BYTES)
//! moves the point; below it nothing changes.

use std::sync::{Arc, Mutex};

/// Largest bound of a discrete logarithm (|x| <= 2^48): above it the search
/// takes hours, and `2 bound + 1` must not wrap. Larger bounds raise ValueError.
pub const MAX_BOUND: u64 = 1 << 48;
/// Baby steps per square root of the range (fewer giant steps for a table built once).
pub const BABY_FACTOR: u64 = 16;
/// Most bytes of one table.
pub const MAX_TABLE_BYTES: usize = 96 << 20;
/// Most tables and bytes of one cache (each of the GT and ristretto255 caches).
pub const CACHE_TABLES: usize = 16;
pub const CACHE_BYTES: usize = 128 << 20;

/// Bytes of a HashMap::with_capacity(m) of entries of `entry` bytes (hashbrown).
pub fn table_bytes(m: u64, entry: usize) -> usize {
    let m = m as usize;
    let buckets = if m < 8 { if m < 4 { 4 } else { 8 } } else { (m * 8 / 7).next_power_of_two() };
    buckets * (entry + 1) + 16
}

/// Most baby steps of a table of entries of `entry` bytes within MAX_TABLE_BYTES.
pub fn max_baby(entry: usize) -> u64 {
    let mut buckets = 8usize;
    while (2 * buckets) * (entry + 1) + 16 <= MAX_TABLE_BYTES {
        buckets *= 2;
    }
    (buckets / 8 * 7) as u64
}

/// Baby steps of a table for |x| <= bound: about BABY_FACTOR sqrt(2 bound + 1)
/// for a cached table, sqrt(2 bound + 1) for a one-shot search (`cached` false),
/// at most max_baby(entry).
pub fn baby_steps(bound: u64, cached: bool, entry: usize) -> u64 {
    let span = 2 * bound + 1; // bound <= MAX_BOUND: no wrap
    let factor = if cached { BABY_FACTOR } else { 1 };
    span.min(((span as f64).sqrt() as u64 + 1) * factor).min(max_baby(entry))
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

    /// (tables, bytes) held now.
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
        assert_eq!(baby_steps(0, true, 40), 1);
        assert_eq!(baby_steps(100, false, 40), 15);
        assert_eq!(max_baby(40), 1_835_008);
        assert_eq!(max_baby(32), 1_835_008);
        assert_eq!(baby_steps(MAX_BOUND, true, 40), 1_835_008);
        assert!(table_bytes(max_baby(40), 40) <= MAX_TABLE_BYTES);
        assert!(table_bytes(max_baby(40) + 1, 40) > MAX_TABLE_BYTES);
    }

    #[test]
    fn table_bytes_match_hashbrown() {
        // the capacity std's HashMap::with_capacity(m) reserves: buckets * 7 / 8 (>= 8 buckets)
        for m in [1usize, 5, 7, 8, 100, 1000, 7168, 7169, 100_000] {
            let cap = std::collections::HashMap::<u64, u64>::with_capacity(m).capacity();
            let buckets = (table_bytes(m as u64, 16) - 16) / 17;
            assert_eq!(cap, if buckets < 8 { buckets - 1 } else { buckets / 8 * 7 }, "m = {m}");
        }
    }
}
