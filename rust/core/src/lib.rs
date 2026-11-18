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
///
/// Two guards against the macOS abort in forked children: [`thread_limit`] (the main one: a forked
/// child starts no threads, so it never waits for any) and [`parallel_for_joined`] (defence in depth:
/// the caller joins its workers instead of parking in the scope, also when a share panics).
pub fn parallel_for<F: Fn(usize) + Sync>(count: usize, threads: usize, f: F) {
    if threads <= 1 || count <= 1 || thread_limit(threads) <= 1 {
        (0..count).for_each(&f);
        return;
    }
    parallel_for_joined(count, threads, f)
}

#[cfg(test)]
mod thread_limit_tests {
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
/// distribution and spawn-failure fallback as [`parallel_for`]; a panic of the
/// caller's share or of a worker reaches the caller after every worker is
/// joined (the caller's first). "Never parks" relies on how std's scope
/// accounts for joined threads (a joined handle releases its thread, so the
/// scope's final wait finds none running); tests/native/test_native_bcfe_fork.py
/// is the guard.
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
        // the caller's share may panic too: catch it, so that every worker is still joined
        // (a handle dropped unjoined would leave the scope to park the caller)
        let mut panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&work)).err();
        // a joined worker is accounted for here, so the scope has nothing left to park for
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

    fn message(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    #[test]
    fn a_panic_of_the_callers_share_reaches_it_after_every_worker() {
        // the workers are slow, so the caller takes some indices; it panics on the first one
        let caller = std::thread::current().id();
        for worker_panics in [false, true] {
            let done: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
            let panicked = AtomicUsize::new(usize::MAX);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parallel_for_joined(64, 4, |i| {
                    if std::thread::current().id() == caller {
                        if panicked.compare_exchange(usize::MAX, i, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            panic!("the caller's share");
                        }
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        if worker_panics && i == 63 {
                            panic!("a worker");
                        }
                    }
                    done[i].fetch_add(1, Ordering::SeqCst);
                })
            }));
            let payload = r.expect_err("the panic reaches the caller");
            assert_eq!(message(&*payload), "the caller's share"); // the caller's panic first
            let p = panicked.load(Ordering::SeqCst);
            assert!(p < 64, "the caller took an index");
            // every other index ran exactly once (but a worker's panicked one): the workers finished
            for (i, d) in done.iter().enumerate() {
                let want = if i == p || (worker_panics && i == 63) { 0 } else { 1 };
                assert_eq!(d.load(Ordering::SeqCst), want, "index {i}");
            }
        }
    }
}
