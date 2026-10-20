//! Randomness of the native calls: the per-call ChaCha20 DRBG (CallDrbg), the
//! OS bytes it is seeded from, and the choice between given bytes (tests) and
//! the DRBG streams for the tensor kernels.

use pyfe4ai_core::osrand;
use pyfe4ai_core::rlwe::Randomness;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Randomness of one native call: ChaCha20 keyed by one 32-byte seed from the OS
/// (or injected by a test), with stream c for chunk c, so that no two chunks --
/// hence no two threads -- share a stream. Seed and generator states are wiped
/// when the call ends; nothing persists across calls.
pub(crate) struct CallDrbg {
    seed: zeroize::Zeroizing<[u8; 32]>,
}

impl CallDrbg {
    pub(crate) fn new(seed: Option<&[u8]>) -> PyResult<Self> {
        let mut s = zeroize::Zeroizing::new([0u8; 32]);
        match seed {
            Some(b) => {
                if b.len() != 32 {
                    return Err(PyValueError::new_err("a DRBG seed has 32 bytes"));
                }
                s.copy_from_slice(b);
            }
            None => osrand::fill(&mut s[..]).map_err(|e| PyOSError::new_err(e.to_string()))?,
        }
        Ok(CallDrbg { seed: s })
    }

    /// The generator of stream `stream` at its start (the caller wipes it, wipe_rng).
    pub(crate) fn rng(&self, stream: u64) -> ChaCha20Rng {
        let mut rng = ChaCha20Rng::from_seed(*self.seed);
        rng.set_stream(stream);
        rng
    }

    /// Bytes of stream `stream` from its start.
    pub(crate) fn fill(&self, stream: u64, out: &mut [u8]) {
        self.fill_at(stream, 0, out);
    }

    /// Bytes of stream `stream` from byte `offset` (a multiple of 4).
    pub(crate) fn fill_at(&self, stream: u64, offset: u64, out: &mut [u8]) {
        debug_assert!(offset % 4 == 0);
        let mut rng = self.rng(stream);
        if offset > 0 {
            rng.set_word_pos((offset / 4) as u128);
        }
        rng.fill_bytes(out);
        wipe_rng(&mut rng);
    }
}

/// ChaCha20Rng holds only integers: zero its bytes before it is dropped.
pub(crate) fn wipe_rng(rng: &mut ChaCha20Rng) {
    unsafe {
        let p = rng as *mut ChaCha20Rng as *mut u8;
        for i in 0..std::mem::size_of::<ChaCha20Rng>() {
            std::ptr::write_volatile(p.add(i), 0);
        }
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// The DRBG output of stream `stream` for `seed` (for the tests' reference stream).
#[pyfunction]
pub(crate) fn drbg_bytes(py: Python<'_>, seed: &[u8], stream: u64, n: usize) -> PyResult<Py<PyBytes>> {
    let d = CallDrbg::new(Some(seed))?;
    let mut out = vec![0u8; n];
    d.fill(stream, &mut out);
    Ok(PyBytes::new(py, &out).unbind())
}

pub(crate) fn os_random(py: Python<'_>, n: usize) -> PyResult<Vec<u8>> {
    let mut b = vec![0u8; n];
    py.detach(|| osrand::fill(&mut b)).map_err(|e| PyOSError::new_err(e.to_string()))?;
    Ok(b)
}

#[pyfunction]
pub(crate) fn random_bytes(py: Python<'_>, n: usize) -> PyResult<Py<PyBytes>> {
    Ok(PyBytes::new(py, &os_random(py, n)?).unbind())
}

/// Given bytes (tests), else the call's DRBG streams.
pub(crate) fn randomness<'a>(rnd: &'a Option<Vec<u8>>, stream: &'a (dyn Fn(usize, u64, &mut [u8]) + Sync)) -> Randomness<'a> {
    match rnd {
        Some(b) => Randomness::Given(b),
        None => Randomness::Stream(stream),
    }
}
