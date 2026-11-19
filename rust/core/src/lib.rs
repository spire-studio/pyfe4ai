//! Kernels of PyFE4AI's lattice schemes, without dependencies, bound to Python
//! by the `pyfe4ai-native` crate (rust/native). Every kernel computes the values of the Python
//! reference implementation (see tests/ and pyfe4ai's tests/native/test_native_backend.py).

pub mod arith;
pub mod bigauss;
pub mod capi;
pub mod codec;
pub mod dense;
pub mod facct;
pub mod facct_wide;
pub mod gauss;
pub mod keccak;
pub mod ntt;
pub mod osrand;
pub mod replicated;
pub mod rlwe;
pub mod rns;
pub mod shamir;
pub mod sha256;
pub mod sha512;
pub mod threshold;

pub use arith::{Modulus, Word, U256};
pub use ntt::{AnyRing, Ring};

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// The process that first ran [`parallel_for`] with several threads (0 until one did).
static THREADED_PID: AtomicU32 = AtomicU32::new(0);

/// The number of threads [`parallel_for`] uses for a request of `threads`: `threads`, or 1 in a
/// process forked (by any means) from a process that had already run it with several threads.
///
/// On macOS the standard library parks a waiting thread on a libdispatch semaphore, and libdispatch
/// refuses to work in the child of a fork from a multi-threaded process: the next park in the child
/// aborts it (SIGTRAP, "BUG IN CLIENT OF LIBDISPATCH: Use-after-free of dispatch_semaphore_t",
/// "crashed on child side of fork pre-exec"). [`parallel_for`] parks its caller while its workers
/// finish, and other waits of the standard library (`Once`, `OnceLock` under contention) park too, so
/// a forked child starts no worker threads at all. The first process to run several threads records
/// its pid here with one compare-and-swap (no wait); a process whose pid differs is a fork of it (or
/// of a fork of it), so a later thread setting there cannot re-enable threads, whatever path the call
/// took (the Python backend or `_rust_native` directly) and whoever forked (`os.fork`, a C extension).
/// A process that never ran several threads records nothing, and its children may use threads.
/// No randomness or other state is cached for this: workers exit before [`parallel_for`] returns.
pub fn thread_limit(threads: usize) -> usize {
    if threads <= 1 {
        return threads;
    }
    let me = std::process::id();
    match THREADED_PID.compare_exchange(0, me, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => threads,
        Err(owner) if owner == me => threads,
        Err(_) => 1,
    }
}

/// What [`thread_limit`] would return for `threads`, without recording this process.
pub fn thread_limit_peek(threads: usize) -> usize {
    let owner = THREADED_PID.load(Ordering::Acquire);
    if threads > 1 && owner != 0 && owner != std::process::id() {
        1
    } else {
        threads
    }
}

/// Run f(0) .. f(count - 1) on up to `threads` scoped threads (the caller's included); with one
/// thread only in a forked child of a process that ran several (see [`thread_limit`]).
pub fn parallel_for<F: Fn(usize) + Sync>(count: usize, threads: usize, f: F) {
    if threads <= 1 || count <= 1 || thread_limit(threads) <= 1 {
        (0..count).for_each(&f);
        return;
    }
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= count {
            break;
        }
        f(i);
    };
    std::thread::scope(|s| {
        for _ in 1..threads.min(count) {
            if std::thread::Builder::new().spawn_scoped(s, &work).is_err() {
                break; // fewer threads; this one finishes the work
            }
        }
        work();
    });
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    #[test]
    fn parallel_for_runs_every_index_once() {
        for (count, threads) in [(0usize, 4usize), (1, 4), (7, 1), (100, 3), (1000, 8), (5, 64)] {
            let seen: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
            parallel_for(count, threads, |i| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1), "{count} {threads}");
        }
    }

    #[test]
    fn the_process_that_runs_threads_keeps_them() {
        parallel_for(64, 4, |_| {});
        assert_eq!(thread_limit_peek(4), 4);
        assert_eq!(thread_limit(4), 4);
        assert_eq!(thread_limit(1), 1);
        assert_eq!(THREADED_PID.load(Ordering::Acquire), std::process::id());
    }
}
