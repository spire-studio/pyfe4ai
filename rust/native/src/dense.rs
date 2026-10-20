//! Dense-LWE MCFE (pyfe4ai.schemes.mcfe.lwe): the sums <key, H(label)> of every
//! label material (pyfe4ai-core `dense`).

use crate::values::materials_of;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Dense-LWE sums of every material; with `limit` (nb big-endian bytes) the
/// blocks are the rejection-sampled candidates of SHAKE with `rate` (136 or 168).
#[pyfunction]
#[pyo3(signature = (materials, nb, count, key, kl, threads, limit=None, rate=136))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_dots(py: Python<'_>, materials: &Bound<'_, PyAny>, nb: usize, count: usize, key: &[u8], kl: usize,
              threads: usize, limit: Option<Vec<u8>>, rate: usize) -> PyResult<Py<PyBytes>> {
    let mats = materials_of(materials)?.unwrap_or_default();
    // checked sizes: count * kl * 8 is the key's length, nb * count the hash stream's
    let key_len = count.checked_mul(kl).and_then(|v| v.checked_mul(8));
    if nb == 0 || nb > 1 << 16 || kl == 0 || key_len != Some(key.len()) || nb.checked_mul(count).is_none()
        || limit.as_ref().is_some_and(|l| l.len() != nb) || (rate != 136 && rate != 168) {
        return Err(PyValueError::new_err("inconsistent dense_dots arguments"));
    }
    if limit.as_ref().is_some_and(|l| l.iter().all(|&b| b == 0)) {
        return Err(PyValueError::new_err("dense_dots: a zero limit accepts nothing"));
    }
    let k: Vec<u64> = key.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let refs: Vec<&[u8]> = mats.iter().map(|v| v.as_slice()).collect();
    let sums = py.detach(|| match &limit {
        None => pyfe4ai_core::dense::dense_dots(&refs, nb, count, &k, kl, threads),
        Some(l) => pyfe4ai_core::dense::dense_dots_rej(&refs, nb, l, rate, count, &k, kl, threads),
    });
    Ok(PyBytes::new(py, &sums.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unbind())
}
