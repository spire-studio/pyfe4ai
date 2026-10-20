//! pyfe4ai._rust_native: the Rust backend of pyfe4ai.backend.
//!
//! The lattice functions back the Ring-LWE and dense-LWE schemes (checked bit
//! for bit against their Python code by the tests); `ec` adds ristretto255 and
//! `pairing` BLS12-381. Values cross the boundary as Python ints, decimal
//! strings (the ciphertext format) or packed little-endian words in bytes; the
//! GIL is released around the kernels.
//!
//! Modules, by the schemes and primitives they serve:
//! * `ring`: the ring context (RingCtx), packing, the NTT, products and linear
//!   combinations; `hashing`: SHAKE-256 and labels hashed into Z_q;
//!   `sampling`: the Gaussian samplers (rounded, FACCT, wide widths);
//! * `mife`: slot keys, encryption of rows (SIFE and MIFE), decryption of the
//!   chunks of a layer; `mcfe`: encryption of a layer, label key terms (MCFE,
//!   dMCFE), offline / online; `threshold`: threshold MCFE (dealing, partial
//!   decryptions, combination); `replicated`: replicated sharing of the LWE
//!   threshold layers; `dense`: dense-LWE MCFE; `rns`: the lattice kernels for
//!   an RNS modulus q = q1 q2;
//! * `ec` (ristretto255), `pairing` and `msm` (BLS12-381), `dlog` (their
//!   discrete-logarithm tables);
//! * shared by the others: `drbg` (the per-call DRBG and OS randomness),
//!   `values` (Python values to and from words mod q) and `secret` (copies of
//!   secret inputs and their wipes).

use pyfe4ai_core::osrand;
use pyo3::prelude::*;

mod dense;
mod dlog;
mod drbg;
mod ec;
mod hashing;
mod mcfe;
mod mife;
mod msm;
mod pairing;
mod replicated;
mod ring;
mod rns;
mod sampling;
mod secret;
mod threshold;
mod values;

#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("compiler", concat!("rustc (pyfe4ai-native ", env!("CARGO_PKG_VERSION"), ", pyo3)"))?;
    d.set_item("implementation", "rust")?;
    d.set_item("os_random", osrand::AVAILABLE)?;
    d.set_item("sampling_randomness", "ChaCha20 per call, seeded from the OS")?;
    d.set_item("hardware_threads", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))?;
    Ok(d.into_any().unbind())
}

/// The classes and functions of the domain modules in one list (this order is
/// the order of the module's namespace), then the ones `ec`, `pairing`, `rns`,
/// `replicated` and `threshold` add with their register functions.
#[pymodule]
fn _rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ring::RingCtx>()?;
    m.add_class::<mife::EncKeyObj>()?;
    m.add_class::<mcfe::PrecomputedObj>()?;
    m.add_class::<sampling::FacctObj>()?;
    m.add_class::<sampling::FacctWideObj>()?;
    for f in [
        wrap_pyfunction!(ring::ctx_new, m)?, wrap_pyfunction!(ring::ctx_new_rns, m)?, wrap_pyfunction!(ring::ctx_info, m)?,
        wrap_pyfunction!(ring::pack, m)?, wrap_pyfunction!(ring::unpack, m)?, wrap_pyfunction!(ring::ntt, m)?,
        wrap_pyfunction!(ring::ntt_mont, m)?, wrap_pyfunction!(ring::poly_mul, m)?, wrap_pyfunction!(ring::lincomb, m)?,
        wrap_pyfunction!(sampling::gauss, m)?, wrap_pyfunction!(hashing::shake256_py, m)?,
        wrap_pyfunction!(hashing::hash_to_zq_py, m)?, wrap_pyfunction!(drbg::random_bytes, m)?,
        wrap_pyfunction!(mife::enc_key, m)?, wrap_pyfunction!(mife::encrypt, m)?, wrap_pyfunction!(mcfe::encrypt_tensor, m)?,
        wrap_pyfunction!(sampling::facct_sample, m)?, wrap_pyfunction!(mife::decrypt_tensor, m)?,
        wrap_pyfunction!(mcfe::pad_terms, m)?, wrap_pyfunction!(mcfe::precompute_tensor, m)?,
        wrap_pyfunction!(mcfe::encrypt_precomputed, m)?, wrap_pyfunction!(mcfe::precomputed_info, m)?,
        wrap_pyfunction!(dense::dense_dots, m)?, wrap_pyfunction!(build_info, m)?, wrap_pyfunction!(drbg::drbg_bytes, m)?,
        wrap_pyfunction!(threshold::share_inputs, m)?, wrap_pyfunction!(threshold::share_partial, m)?,
        wrap_pyfunction!(threshold::share_combine, m)?, wrap_pyfunction!(threshold::shamir_share, m)?,
        wrap_pyfunction!(sampling::gauss_bigint, m)?, wrap_pyfunction!(sampling::gauss_bigint_approx, m)?,
        wrap_pyfunction!(sampling::facct_draw, m)?, wrap_pyfunction!(sampling::facct_draw_wide, m)?,
    ] {
        m.add_function(f)?;
    }
    ec::register(m)?;
    pairing::register(m)?;
    rns::register(m)?;
    replicated::register(m)?;
    threshold::register(m)?;
    Ok(())
}
