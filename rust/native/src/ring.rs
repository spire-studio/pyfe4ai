//! The ring Z_q[x] / (x^n + 1) of the Ring-LWE schemes: its context (RingCtx;
//! the functions here dispatch an rns::RnsCtx, q = q1 q2, to `rns`), values
//! packed as words, the NTT, products, and the linear combinations of the key
//! derivations (lincomb).

use crate::rns;
use crate::secret::{secret_byte_rows, wipe_byte_rows, wipe_words};
use crate::threshold;
use crate::values::{bytes_to_words, dec_list, py_index, read_values, u256_arg, word_pyint, words_to_bytes};
use pyfe4ai_core::{with_ring, AnyRing, Ring, Word};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::sync::Arc;

/// A ring Z_q[x] / (x^n + 1) with its NTT tables (ctx_new).
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
pub(crate) struct RingCtx {
    ring: Arc<AnyRing>,
    pub(crate) q: Py<PyAny>,
}

pub(crate) fn get_ring(ctx: &Bound<'_, RingCtx>) -> Arc<AnyRing> {
    ctx.get().ring.clone()
}

#[pyfunction]
pub(crate) fn ctx_new(py: Python<'_>, q: &Bound<'_, PyAny>, n: usize, root: &Bound<'_, PyAny>) -> PyResult<RingCtx> {
    let (qv, rv) = (u256_arg(q)?, u256_arg(root)?);
    let ring = py.detach(|| AnyRing::new(qv, n, rv)).map_err(PyValueError::new_err)?;
    Ok(RingCtx { ring: Arc::new(ring), q: py_index(q)?.unbind() })
}

/// A ring for a composite q = primes[0] ... primes[k-1] (AnyRing::new_rns: one
/// 64-bit NTT per prime under the standard-value kernels; 2 <= k <= 4 primes
/// below 2^62, roots the residues of the CRT root). The kernels of a RingCtx
/// run on it unchanged, with the values of a ring mod q.
#[pyfunction]
pub(crate) fn ctx_new_rns(py: Python<'_>, q: &Bound<'_, PyAny>, n: usize, primes: Vec<u64>, roots: Vec<u64>) -> PyResult<RingCtx> {
    let qv = u256_arg(q)?;
    let ring = py.detach(|| AnyRing::new_rns(qv, n, &primes, &roots)).map_err(PyValueError::new_err)?;
    Ok(RingCtx { ring: Arc::new(ring), q: py_index(q)?.unbind() })
}

#[pyfunction]
pub(crate) fn ctx_info(py: Python<'_>, ctx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return rns::ctx_info(py, c);
    }
    let r = &ctx.cast::<RingCtx>()?.get().ring;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("word_bits", r.word_bits())?;
    d.set_item("n", r.n())?;
    d.set_item("q_bits", r.qbits())?;
    let primes: Option<Vec<u64>> = with_ring!(&**r, rg => rg.rns.as_ref().map(|b| b.rings.iter().map(|x| x.q).collect()));
    if let Some(p) = primes {
        d.set_item("rns", p)?;
    }
    Ok(d.into_any().unbind())
}

#[pyfunction]
pub(crate) fn pack(py: Python<'_>, ctx: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return rns::pack(py, c, values);
    }
    let ctx = ctx.cast::<RingCtx>()?;
    let q = ctx.get().q.bind(py);
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        // binary key-share polynomials: ceil(bits(q) / 8)-byte words, each below q
        if let Ok(b) = values.cast::<PyBytes>() {
            let wb = threshold::value_bytes_of(&r.m);
            if b.as_bytes().len() % wb != 0 {
                return Err(PyValueError::new_err("not a whole number of values"));
            }
            let v = threshold::SecretWords(threshold::read_poly(values, b.as_bytes().len() / wb, q, &r.m)?);
            return Ok(words_to_bytes(py, &v));
        }
        // the values may be secret (key shares on their way to ntt_mont): wiped after packing
        let v = threshold::SecretWords(read_values(values, None, q, &r.m, false)?);
        Ok(words_to_bytes(py, &v))
    })
}

#[pyfunction]
#[pyo3(signature = (ctx, data, as_digits=false))]
pub(crate) fn unpack(py: Python<'_>, ctx: &Bound<'_, PyAny>, data: &[u8], as_digits: bool) -> PyResult<Py<PyList>> {
    if ctx.cast::<rns::RnsCtx>().is_ok() {
        return rns::unpack(py, data, as_digits);
    }
    let ring = get_ring(ctx.cast::<RingCtx>()?);
    with_ring!(&*ring, r => unpack_w(py, r, data, as_digits))
}

fn unpack_w<W: Word>(py: Python<'_>, _r: &Ring<W>, data: &[u8], as_digits: bool) -> PyResult<Py<PyList>> {
    let size = W::BITS as usize / 8;
    if data.len() % size != 0 {
        return Err(PyValueError::new_err("length is not a multiple of the word size"));
    }
    let w: Vec<W> = data.chunks_exact(size).map(W::read_le).collect();
    if as_digits {
        return Ok(dec_list(py, &w)?.unbind());
    }
    let ints = w.iter().map(|&v| word_pyint(py, v)).collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, ints)?.unbind())
}

#[pyfunction]
#[pyo3(signature = (ctx, data, inverse=false))]
pub(crate) fn ntt(py: Python<'_>, ctx: &Bound<'_, PyAny>, data: &[u8], inverse: bool) -> PyResult<Py<PyBytes>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return rns::ntt(py, c, data, inverse);
    }
    let ring = get_ring(ctx.cast::<RingCtx>()?);
    with_ring!(&*ring, r => {
        let mut x = bytes_to_words(data, r.n)?;
        let mut y = x.clone();
        // NTT-domain values cross as the NTT mod q (an RNS ring holds residues)
        if inverse {
            r.ntt_from_std(&mut x);
            r.inverse(&x, &mut y)
        } else {
            r.forward(&x, &mut y);
            r.ntt_to_std(&mut y);
        }
        Ok(words_to_bytes(py, &y))
    })
}

#[pyfunction]
pub(crate) fn ntt_mont(py: Python<'_>, ctx: &Bound<'_, PyAny>, data: &[u8]) -> PyResult<Py<PyBytes>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return rns::ntt_mont(py, c, data);
    }
    let ring = get_ring(ctx.cast::<RingCtx>()?);
    with_ring!(&*ring, r => {
        // the input is a key share (the NTT(sk_y) cache of the threshold servers): both copies wiped
        let x = threshold::SecretWords(bytes_to_words(data, r.n)?);
        let mut y = threshold::SecretWords(x.0.clone());
        r.forward_mont(&x, &mut y.0);
        Ok(words_to_bytes(py, &y))
    })
}

#[pyfunction]
pub(crate) fn poly_mul(py: Python<'_>, ctx: &Bound<'_, PyAny>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return rns::poly_mul(py, c, a, b);
    }
    let ring = get_ring(ctx.cast::<RingCtx>()?);
    with_ring!(&*ring, r => {
        let (x, y) = (bytes_to_words(a, r.n)?, bytes_to_words(b, r.n)?);
        Ok(words_to_bytes(py, &r.poly_mul(&x, &y)))
    })
}

/// sum_i coeffs[i] rows[i] mod q as decimal strings. The copies of the rows and
/// the accumulator are wiped (rows may be secret key pieces).
#[pyfunction]
pub(crate) fn lincomb(py: Python<'_>, ctx: &Bound<'_, PyAny>, rows: &Bound<'_, PyAny>, coeffs: &Bound<'_, PyAny>)
    -> PyResult<Py<PyList>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return rns::lincomb(py, c, rows, coeffs);
    }
    let ctx = ctx.cast::<RingCtx>()?;
    let q = ctx.get().q.bind(py);
    let ring = get_ring(ctx);
    let mut rows = secret_byte_rows(rows)?;
    let out = with_ring!(&*ring, r => {
        (|| {
            let ys = read_values(coeffs, None, q, &r.m, false)?;
            if rows.len() != ys.len() {
                return Err(PyValueError::new_err("one coefficient per row"));
            }
            let mut acc = vec![Word::ZERO; r.n];
            for (row, &y) in rows.iter().zip(ys.iter()) {
                let mut v = bytes_to_words(row, r.n)?;
                let ym = r.m.to_mont(y);
                for i in 0..r.n {
                    acc[i] = r.m.add(acc[i], r.m.mul(v[i], ym));
                }
                wipe_words(&mut v);
            }
            let out = dec_list(py, &acc).map(|l| l.unbind());
            wipe_words(&mut acc);
            out
        })()
    });
    wipe_byte_rows(&mut rows);
    out
}
