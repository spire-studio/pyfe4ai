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

use std::sync::atomic::{AtomicUsize, Ordering};

/// Run f(0) .. f(count - 1) on up to `threads` scoped threads (the caller's included).
///
/// Delegates to [`parallel_for_joined`]: the caller joins its workers and never parks, so a process
/// may fork after a multi-threaded call and run kernels with several threads in the child (macOS:
/// parking there aborted the child, see [`parallel_for_joined`]).
pub fn parallel_for<F: Fn(usize) + Sync>(count: usize, threads: usize, f: F) {
    parallel_for_joined(count, threads, f)
}

/// [`parallel_for`] with the caller joining every worker it spawned instead
/// of waiting for them in `std::thread::scope`, which parks it until they are
/// done. On macOS the standard library parks a thread on a dispatch semaphore
/// whose Mach port is created at the first slow wait; a forked child inherits
/// the semaphore but not the port, and the next park of the forking thread in
/// the child aborts the process (SIGTRAP, "BUG IN CLIENT OF LIBDISPATCH:
/// Use-after-free of dispatch_semaphore_t"). [`parallel_for`] parks its caller
/// whenever the workers finish after it does; this function never parks it
/// (`JoinHandle::join` waits in `pthread_join`), so a process may fork after
/// it and run it again in the child with several threads. The same work
/// distribution and spawn-failure fallback as [`parallel_for`].
pub fn parallel_for_joined<F: Fn(usize) + Sync>(count: usize, threads: usize, f: F) {
    if threads <= 1 || count <= 1 {
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
        let mut workers = Vec::with_capacity(threads.min(count) - 1);
        for _ in 1..threads.min(count) {
            match std::thread::Builder::new().spawn_scoped(s, &work) {
                Ok(handle) => workers.push(handle),
                Err(_) => break, // fewer threads; this one finishes the work
            }
        }
        work();
        // a joined worker is accounted for here, so the scope has nothing left to park for
        let mut panic = None;
        for handle in workers {
            if let Err(payload) = handle.join() {
                panic.get_or_insert(payload);
            }
        }
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
    });
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    #[test]
    fn joined_parallel_for_runs_every_index_once() {
        for (count, threads) in [(0usize, 4usize), (1, 4), (7, 1), (100, 3), (1000, 8), (5, 64)] {
            let seen: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
            parallel_for_joined(count, threads, |i| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1), "{count} {threads}");
        }
    }

    #[test]
    fn a_worker_panic_reaches_the_caller() {
        let r = std::panic::catch_unwind(|| parallel_for_joined(64, 4, |i| assert!(i != 40, "index 40")));
        assert!(r.is_err());
    }
}
