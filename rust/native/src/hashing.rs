//! SHAKE-256 and the label hash into Z_q of pyfe4ai.primitives.hashing.

use crate::ring::{get_ring, RingCtx};
use crate::rns;
use crate::secret::wipe_words;
use crate::values::words_to_bytes;
use pyfe4ai_core::keccak::{hash_to_zq_mode, shake256, HashMode};
use pyfe4ai_core::{with_ring, Word};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction(name = "shake256")]
pub(crate) fn shake256_py(py: Python<'_>, data: &[u8], length: usize) -> PyResult<Py<PyBytes>> {
    let mut out = vec![0u8; length];
    py.detach(|| shake256(data, &mut out));
    Ok(PyBytes::new(py, &out).unbind())
}

pub(crate) fn hash_mode(code: u8) -> PyResult<HashMode> {
    HashMode::from_code(code).ok_or_else(|| PyValueError::new_err("unknown label-hash mode"))
}

/// pyfe4ai.primitives.hashing.hash_to_zq_vector of the material (mode: 0 shake256-wide,
/// 1 shake256-rejection, 2 shake128-rejection), packed words.
#[pyfunction(name = "hash_to_zq")]
#[pyo3(signature = (ctx, material, count, hash_mode=0))]
pub(crate) fn hash_to_zq_py(py: Python<'_>, ctx: &Bound<'_, PyAny>, material: &[u8], count: usize, hash_mode: u8)
    -> PyResult<Py<PyBytes>> {
    let mode = self::hash_mode(hash_mode)?;
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return Ok(rns::hash_to_zq(py, c, material, count, mode));
    }
    let ring = get_ring(ctx.cast::<RingCtx>()?);
    with_ring!(&*ring, r => {
        let mut out = vec![Word::ZERO; count];
        py.detach(|| hash_to_zq_mode(&r.m, material, mode, &mut out));
        let bytes = words_to_bytes(py, &out);
        wipe_words(&mut out);
        Ok(bytes)
    })
}
