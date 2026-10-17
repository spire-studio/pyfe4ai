//! Kernels of PyFE4AI's lattice schemes, without dependencies, bound to Python
//! by the `pyfe4ai-native` crate (rust/native). Every kernel computes the values of the Python
//! reference implementation (see tests/ and pyfe4ai's tests/test_native_backend.py).

pub mod arith;
pub mod capi;
pub mod codec;
pub mod dense;
pub mod facct;
pub mod gauss;
pub mod keccak;
pub mod ntt;
pub mod osrand;
pub mod rlwe;
pub mod shamir;
pub mod sha256;
pub mod sha512;

pub use arith::{Modulus, Word, U256};
pub use ntt::{AnyRing, Ring};

use std::sync::atomic::{AtomicUsize, Ordering};

/// Run f(0) .. f(count - 1) on up to `threads` scoped threads (the caller's included).
pub fn parallel_for<F: Fn(usize) + Sync>(count: usize, threads: usize, f: F) {
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
        for _ in 1..threads.min(count) {
            if std::thread::Builder::new().spawn_scoped(s, &work).is_err() {
                break; // fewer threads; this one finishes the work
            }
        }
        work();
    });
}
