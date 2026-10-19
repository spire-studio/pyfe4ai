//! The threshold Ring-LWE MCFE's binary paths (pyfe4ai-core `threshold`):
//! partial decryptions of format-2 ciphertexts (`ct_format = "binary"`
//! public parameters) with outputs as bytes, their combination, and the
//! uniform flooding noise, which the JSON path's share_partial draws too.
//! Values of key shares may be decimal strings, ints or fixed-width bytes
//! (read_poly); the noise, the key part and the caller's noise material
//! are wiped before they are freed.

use super::{delta_mod, get_ring, index_mod, py_index, read_rows, read_values, RingCtx};
use pyfe4ai_core::rlwe::{wipe_bytes, wipe_words};
use pyfe4ai_core::threshold::{self, value_bytes};
use pyfe4ai_core::{with_ring, Modulus, Word};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// n values mod q: a sequence as read_values reads it, or bytes of n
/// ceil(bits(q) / 8)-byte little-endian words, each below q.
pub(crate) fn read_poly<W: Word>(ob: &Bound<'_, PyAny>, n: usize, q: &Bound<'_, PyAny>, m: &Modulus<W>)
    -> PyResult<Vec<W>> {
    if let Ok(b) = ob.cast::<PyBytes>() {
        let mut out = vec![W::ZERO; n];
        if threshold::read_strict(b.as_bytes(), m.q, value_bytes(m.bits), &mut out).is_none() {
            wipe_words(&mut out);
            return Err(PyValueError::new_err(format!(
                "expected {} values below q as {}-byte words", n, value_bytes(m.bits))));
        }
        return Ok(out);
    }
    read_values(ob, Some(n), q, m, false)
}

/// The flooding noise of a partial decryption: `count` values of the given
/// list, or the uniform noise of `flood = (bits, material)` drawn here; the
/// material is wiped.
pub(crate) fn noise_values<W: Word>(noise: Option<&Bound<'_, PyAny>>, flood: Option<(u32, Vec<u8>)>, count: usize,
                                    q: &Bound<'_, PyAny>, m: &Modulus<W>) -> PyResult<Vec<W>> {
    match (noise, flood) {
        (None, Some((bits, mut material))) => {
            let mut out = vec![W::ZERO; count];
            let res = threshold::uniform_flooding(m, bits, &material, &mut out);
            wipe_bytes(&mut material);
            res.map_err(PyValueError::new_err)?;
            Ok(out)
        }
        (Some(list), None) => read_values(list, Some(count), q, m, false),
        _ => Err(PyValueError::new_err("give the noise or the uniform flooding (bits, material), not both")),
    }
}

fn values_bytes<W: Word>(py: Python<'_>, vals: &[W], wb: usize) -> Py<PyBytes> {
    PyBytes::new_with(py, vals.len() * wb, |out| {
        for (o, v) in out.chunks_exact_mut(wb).zip(vals) {
            v.write_le(o);
        }
        Ok(())
    })
    .expect("allocating bytes")
    .unbind()
}

/// ring_lwe_threshold.uniform_flooding_poly(bits, count, seed) mod q, for the
/// material domain || len(seed) || seed (tests compare it with the Python code).
#[pyfunction]
fn uniform_flooding(py: Python<'_>, ctx: &Bound<'_, RingCtx>, bits: u32, material: Vec<u8>, count: usize)
    -> PyResult<Py<PyList>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let mut v = noise_values(None, Some((bits, material)), count, &q, &r.m)?;
        let items = v.iter().map(|&x| super::word_pyint(py, x)).collect::<PyResult<Vec<_>>>();
        wipe_words(&mut v);
        Ok(PyList::new(py, items?)?.unbind())
    })
}

/// ThresholdMCFERingLWE.share_decrypt of one chunk of format-2 ciphertexts
/// (`cts`, one per client in the order of `ys` and `sk_hats`): (w, c_y, k), w
/// and c_y the first k values as ceil(bits(q) / 8)-byte words. `z` is the
/// key share's term (n values: a sequence or bytes), the noise k values (or
/// `flood` for the uniform noise drawn here). ValueError("malformed
/// ciphertext", i) for a ciphertext that is not a strict format-2 one of
/// these parameters, and for different batch lengths.
#[pyfunction]
#[pyo3(signature = (ctx, cts, eta, ys, sk_hats, z, delta, lagrange, noise=None, flood=None))]
#[allow(clippy::too_many_arguments)]
fn share_partial_bin(py: Python<'_>, ctx: &Bound<'_, RingCtx>, cts: Vec<Vec<u8>>, eta: usize, ys: &Bound<'_, PyAny>,
                     sk_hats: Vec<Vec<u8>>, z: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>,
                     lagrange: &Bound<'_, PyAny>, noise: Option<&Bound<'_, PyAny>>, flood: Option<(u32, Vec<u8>)>)
    -> PyResult<(Py<PyBytes>, Py<PyBytes>, usize)> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    if cts.is_empty() || cts.len() != sk_hats.len() || eta == 0 {
        return Err(PyValueError::new_err("one ciphertext and key share per client"));
    }
    with_ring!(&*ring, r => {
        let (n, m) = (r.n, &r.m);
        let mut parsed = Vec::with_capacity(cts.len());
        for (i, ct) in cts.iter().enumerate() {
            match threshold::read_ciphertext(ct, m.q, eta, r.logn) {
                Some(p) => parsed.push(p),
                None => return Err(PyValueError::new_err(("malformed ciphertext", i))),
            }
        }
        let k = parsed[0].0;
        if let Some(i) = parsed.iter().position(|p| p.0 != k) {
            return Err(PyValueError::new_err(("malformed ciphertext", i)));
        }
        let hats: Vec<Vec<_>> = sk_hats.iter().map(|b| super::bytes_to_words(b, n)).collect::<PyResult<_>>()?;
        let y_rows = read_rows(ys, Some(eta), &q, m)?;
        if y_rows.len() != parsed.len() {
            return Err(PyValueError::new_err("one weight row per client"));
        }
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
        let mut zv = read_poly(z, n, &q, m)?;
        let mut ev = noise_values(noise, flood, k, &q, m)?;
        let (d, l) = (delta_mod(delta, &q, m)?, index_mod(lagrange, &q, m)?);
        let (mut w, mut cy) = (vec![Word::ZERO; k], vec![Word::ZERO; k]);
        py.detach(|| {
            let c0: Vec<&[_]> = parsed.iter().map(|p| p.1.as_slice()).collect();
            let c1: Vec<&[_]> = parsed.iter().map(|p| p.2.as_slice()).collect();
            let hr: Vec<&[_]> = hats.iter().map(|v| v.as_slice()).collect();
            threshold::share_partial_k(r, eta, &c0, &c1, &ym, &hr, &zv, d, l, &ev, &mut w, &mut cy);
        });
        wipe_words(&mut zv);
        wipe_words(&mut ev);
        let wb = value_bytes(m.bits);
        Ok((values_bytes(py, &w, wb), values_bytes(py, &cy, wb), k))
    })
}

/// ThresholdMCFERingLWE._combine_group of binary partial decryptions: the k
/// decoded values (ints, None where |mu| > bound), or None if one does not fit
/// an int64. ValueError unless c_y and every w are k strict values.
#[pyfunction]
fn share_combine_bin(py: Python<'_>, ctx: &Bound<'_, RingCtx>, c_y: &[u8], ws: Vec<Vec<u8>>, delta: &Bound<'_, PyAny>,
                     bound: &Bound<'_, PyAny>, k: usize) -> PyResult<Option<Py<PyList>>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let bound_v: i128 = match py_index(bound)?.extract::<i64>() {
        Ok(b) => b as i128,
        Err(_) => if py_index(bound)?.gt(0)? { 1i128 << 126 } else { -1 },
    };
    with_ring!(&*ring, r => {
        let m = &r.m;
        let wb = value_bytes(m.bits);
        if k < 1 || k > r.n {
            return Err(PyValueError::new_err("malformed partial decryption"));
        }
        let mut cy = vec![Word::ZERO; k];
        threshold::read_strict(c_y, m.q, wb, &mut cy).ok_or_else(|| PyValueError::new_err("malformed partial decryption"))?;
        let mut wv = Vec::with_capacity(ws.len());
        for w in ws.iter() {
            let mut v = vec![Word::ZERO; k];
            threshold::read_strict(w, m.q, wb, &mut v).ok_or_else(|| PyValueError::new_err("malformed partial decryption"))?;
            wv.push(v);
        }
        let d = delta_mod(delta, &q, m)?;
        let (mut out, mut none) = (vec![0i64; k], vec![0u8; k]);
        let wr: Vec<&[_]> = wv.iter().map(|v| v.as_slice()).collect();
        if !py.detach(|| threshold::combine(m, &cy, &wr, d, bound_v, &mut out, &mut none)) {
            return Ok(None);
        }
        let items = out.iter().zip(none.iter())
            .map(|(&v, &nn)| if nn == 1 { Ok(py.None().into_bound(py)) } else { Ok(v.into_pyobject(py)?.into_any()) })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some(PyList::new(py, items)?.unbind()))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniform_flooding, m)?)?;
    m.add_function(wrap_pyfunction!(share_partial_bin, m)?)?;
    m.add_function(wrap_pyfunction!(share_combine_bin, m)?)?;
    Ok(())
}
