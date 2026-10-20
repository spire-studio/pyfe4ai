//! Python values and words mod q: ints and decimal strings in (read_values,
//! read_rows, index_mod, ...), packed little-endian words and decimal strings
//! out (words_to_bytes, dec_list, DecBuf), and the gathering of ciphertext
//! strings for the parsers that run without the GIL.

use crate::secret::wipe_word_rows;
use pyfe4ai_core::rlwe;
use pyfe4ai_core::{Modulus, Word, U256};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList, PyString, PyTuple};

pub(crate) fn u256_arg(ob: &Bound<'_, PyAny>) -> PyResult<U256> {
    let v = py_index(ob)?;
    let err = || PyValueError::new_err("needs a non-negative 256-bit integer");
    if v.lt(0)? {
        return Err(err());
    }
    let b = v.call_method1(intern!(ob.py(), "to_bytes"), (32, intern!(ob.py(), "little"))).map_err(|_| err())?;
    Ok(U256::read_le(b.cast::<PyBytes>()?.as_bytes()))
}

pub(crate) fn py_index<'py>(ob: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // operator.index(ob)
    unsafe { Bound::from_owned_ptr_or_err(ob.py(), pyo3::ffi::PyNumber_Index(ob.as_ptr())) }
}

/// A Python int in [0, 2^BITS) as a word.
fn pyint_word<W: Word>(v: &Bound<'_, PyAny>) -> PyResult<W> {
    if W::BITS <= 128 {
        return Ok(W::from_u128(v.extract::<u128>()?));
    }
    let b = v.call_method1(intern!(v.py(), "to_bytes"), (W::BITS / 8, intern!(v.py(), "little")))?;
    Ok(W::read_le(b.cast::<PyBytes>()?.as_bytes()))
}

/// A word as a Python int.
pub(crate) fn word_pyint<'py, W: Word>(py: Python<'py>, v: W) -> PyResult<Bound<'py, PyAny>> {
    if W::BITS <= 128 {
        return Ok(v.to_u128().into_pyobject(py)?.into_any());
    }
    let mut b = vec![0u8; (W::BITS / 8) as usize];
    v.write_le(&mut b);
    py.get_type::<PyInt>().call_method1(intern!(py, "from_bytes"), (PyBytes::new(py, &b), intern!(py, "little")))
}

/// int(str(ob)) % q (the semantics of canonical_ring_ciphertext).
pub(crate) fn str_int_mod<W: Word>(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<W> {
    let int = ob.py().import("builtins")?.getattr("int")?;
    pyint_word(&int.call1((ob.str()?,))?.rem(q)?)
}

/// ob % q for an exact int, without raising (None: not an exact int, a negative
/// value below -2^63, or a positive one Python must reduce).
pub(crate) fn int_mod_fast<W: Word>(ob: &Bound<'_, PyAny>, m: &Modulus<W>) -> Option<W> {
    let i = ob.cast_exact::<PyInt>().ok()?;
    let mut overflow: std::os::raw::c_int = 0;
    let v = unsafe { pyo3::ffi::PyLong_AsLongLongAndOverflow(i.as_ptr(), &mut overflow) };
    match overflow {
        0 => Some(m.reduce_i128(v as i128)),
        // fails (and is dropped) only at 2^BITS (2^128 for u64 words) and beyond
        1 if W::BITS <= 128 => i.extract::<u128>().ok().map(|x| m.reduce_u128(x)),
        1 => pyint_word::<W>(i.as_any()).ok().map(|x| m.reduce(x)),
        _ => None,
    }
}

/// operator.index(ob) % q.
pub(crate) fn index_mod<W: Word>(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, m: &Modulus<W>) -> PyResult<W> {
    if let Some(x) = int_mod_fast(ob, m) {
        return Ok(x);
    }
    let v = py_index(ob)?;
    if let Some(x) = int_mod_fast(&v, m) {
        return Ok(x);
    }
    pyint_word(&v.rem(q)?)
}

/// Delta = floor(q / p) mod q, refused if 0 (a ValueError rather than a division by zero).
pub(crate) fn delta_mod<W: Word>(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, m: &Modulus<W>) -> PyResult<W> {
    let d = index_mod(ob, q, m)?;
    if d == W::ZERO {
        return Err(PyValueError::new_err("Delta must be nonzero mod q"));
    }
    Ok(d)
}

/// Values of a sequence mod q: decimal strings by the fast path, other strings
/// as int(s), anything else by __index__ (str_semantics: int(str(v)) for all).
pub(crate) fn read_values<W: Word>(seq: &Bound<'_, PyAny>, n: Option<usize>, q: &Bound<'_, PyAny>, m: &Modulus<W>,
                        str_semantics: bool) -> PyResult<Vec<W>> {
    let one = |ob: &Bound<'_, PyAny>| -> PyResult<W> {
        if let Ok(s) = ob.cast::<PyString>() {
            if let Some(v) = s.to_str().ok().and_then(|t| W::parse_dec(t.as_bytes(), m.q)) {
                return Ok(v);
            }
            return str_int_mod(ob, q);
        }
        if str_semantics { str_int_mod(ob, q) } else { index_mod(ob, q, m) }
    };
    let check = |len: usize| match n {
        Some(n) if n != len => Err(PyValueError::new_err(format!("expected {} values", n))),
        _ => Ok(()),
    };
    // the values may be secret (x + u, pads): on a malformed value the ones
    // parsed so far are wiped (reserved up front, so no reallocation leaves copies)
    let mut out: Vec<W> = Vec::new();
    let res = (|| -> PyResult<()> {
        if let Ok(list) = seq.cast::<PyList>() {
            check(list.len())?;
            out.reserve_exact(list.len());
            for ob in list.iter() {
                out.push(one(&ob)?);
            }
            return Ok(());
        }
        let items: Vec<Bound<'_, PyAny>> = seq.try_iter()?.collect::<PyResult<_>>()?;
        check(items.len())?;
        out.reserve_exact(items.len());
        for ob in items.iter() {
            out.push(one(ob)?);
        }
        Ok(())
    })();
    match res {
        Ok(()) => Ok(out),
        Err(e) => {
            rlwe::wipe_words(&mut out);
            Err(e)
        }
    }
}

/// Rows of values (each of n values, or any length); on a malformed row the
/// rows parsed so far are wiped (they may be secret: x + u, pads).
pub(crate) fn read_rows<W: Word>(rows: &Bound<'_, PyAny>, n: Option<usize>, q: &Bound<'_, PyAny>, m: &Modulus<W>)
    -> PyResult<Vec<Vec<W>>> {
    let mut out: Vec<Vec<W>> = Vec::new();
    let res = (|| -> PyResult<()> {
        for row in rows.try_iter()? {
            out.push(read_values(&row?, n, q, m, false)?);
        }
        Ok(())
    })();
    match res {
        Ok(()) => Ok(out),
        Err(e) => {
            wipe_word_rows(&mut out);
            Err(e)
        }
    }
}

// Words cross the boundary as packed native-endian (little-endian) bytes.
pub(crate) fn words_to_bytes<W: Word>(py: Python<'_>, w: &[W]) -> Py<PyBytes> {
    let wb = W::BITS as usize / 8;
    PyBytes::new_with(py, w.len() * wb, |out| {
        for (o, v) in out.chunks_exact_mut(wb).zip(w) {
            v.write_le(o);
        }
        Ok(())
    })
    .expect("allocating bytes")
    .unbind()
}

pub(crate) fn bytes_to_words<W: Word>(b: &[u8], count: usize) -> PyResult<Vec<W>> {
    let wb = W::BITS as usize / 8;
    if b.len() != count * wb {
        return Err(PyValueError::new_err(format!("expected {} bytes of words", count * wb)));
    }
    Ok(b.chunks_exact(wb).map(W::read_le).collect())
}

/// A str of ASCII bytes, without UTF-8 decoding.
fn ascii_str<'py>(py: Python<'py>, b: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    debug_assert!(b.is_ascii());
    unsafe {
        let o = pyo3::ffi::PyUnicode_New(b.len() as pyo3::ffi::Py_ssize_t, 127);
        if o.is_null() {
            return Err(PyErr::fetch(py));
        }
        std::ptr::copy_nonoverlapping(b.as_ptr(), pyo3::ffi::PyUnicode_1BYTE_DATA(o), b.len());
        Ok(Bound::from_owned_ptr(py, o))
    }
}

pub(crate) fn dec_list<'py, W: Word>(py: Python<'py>, vals: &[W]) -> PyResult<Bound<'py, PyList>> {
    let mut buf = Vec::with_capacity(80);
    let items = vals
        .iter()
        .map(|v| {
            buf.clear();
            v.format_dec(&mut buf);
            ascii_str(py, &buf)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// Pre-formatted decimal strings of one polynomial (offsets into one buffer).
pub(crate) struct DecBuf {
    pub(crate) text: Vec<u8>,
    off: Vec<u32>,
}

impl DecBuf {
    pub(crate) fn new<W: Word>(vals: &[W]) -> Self {
        let mut text = Vec::with_capacity(vals.len() * (W::BITS as usize * 3 / 10 + 2));
        let mut off = Vec::with_capacity(vals.len() + 1);
        for v in vals {
            off.push(text.len() as u32);
            v.format_dec(&mut text);
        }
        off.push(text.len() as u32);
        DecBuf { text, off }
    }
    pub(crate) fn list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items = self
            .off
            .windows(2)
            .map(|w| ascii_str(py, &self.text[w[0] as usize..w[1] as usize]))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }
}

pub(crate) fn materials_of(mats: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Vec<u8>>>> {
    if mats.is_none() {
        return Ok(None);
    }
    let v: Vec<Vec<u8>> = mats
        .try_iter()?
        .map(|m| {
            let m = m?;
            m.cast::<PyBytes>().map(|b| b.as_bytes().to_vec()).map_err(|_| PyTypeError::new_err("pad materials must be bytes"))
        })
        .collect::<PyResult<_>>()?;
    Ok(Some(v))
}

/// An int64 buffer (numpy array, bytes, ...) as a vector.
pub(crate) fn int64_buffer(x: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let b: Vec<u8> = if let Ok(b) = x.cast::<PyBytes>() {
        b.as_bytes().to_vec()
    } else {
        x.call_method0("tobytes")?.extract()?
    };
    if b.len() % 8 != 0 || b.is_empty() {
        return Err(PyValueError::new_err("x must be a non-empty int64 buffer"));
    }
    Ok(b.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect())
}

/// One gathered ciphertext value list: plain decimal strings copied, others for Python.
pub(crate) struct GText {
    pub(crate) text: Vec<u8>,
    pub(crate) pos: Vec<(u32, u32)>,
}

pub(crate) struct Gathered<'py> {
    pub(crate) t: GText,
    pub(crate) slow: Vec<(usize, Bound<'py, PyAny>)>,
}

pub(crate) const SLOW: u32 = u32::MAX;

/// The values of a list or tuple of n items; ASCII strings of at most max_len
/// characters are copied for the parser (the rest go to Python).
pub(crate) fn gather<'py>(seq: &Bound<'py, PyAny>, n: usize, max_len: usize) -> Option<Gathered<'py>> {
    let items: Vec<Bound<'py, PyAny>> = if let Ok(l) = seq.cast::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = seq.cast::<PyTuple>() {
        t.iter().collect()
    } else {
        return None;
    };
    if items.len() != n {
        return None;
    }
    let mut g = Gathered { t: GText { text: Vec::with_capacity(n * 24), pos: Vec::with_capacity(n) }, slow: Vec::new() };
    for (i, ob) in items.into_iter().enumerate() {
        let p = ob.as_ptr();
        let plain = unsafe {
            pyo3::ffi::PyUnicode_CheckExact(p) != 0 && pyo3::ffi::PyUnicode_IS_ASCII(p) != 0
                && (pyo3::ffi::PyUnicode_GET_LENGTH(p) as usize) <= max_len
        };
        if plain {
            let t = unsafe {
                std::slice::from_raw_parts(pyo3::ffi::PyUnicode_1BYTE_DATA(p) as *const u8,
                                           pyo3::ffi::PyUnicode_GET_LENGTH(p) as usize)
            };
            g.t.pos.push((g.t.text.len() as u32, t.len() as u32));
            g.t.text.extend_from_slice(t);
        } else {
            g.t.pos.push((SLOW, 0));
            g.slow.push((i, ob));
        }
    }
    Some(g)
}

impl<'py> Gathered<'py> {
    /// Value i as a Python object (the original, or its text).
    pub(crate) fn object(&self, py: Python<'py>, i: usize) -> Bound<'py, PyAny> {
        match self.slow.iter().find(|(k, _)| *k == i) {
            Some((_, o)) => o.clone().into_any(),
            None => {
                let (a, l) = self.t.pos[i];
                PyString::new(py, std::str::from_utf8(&self.t.text[a as usize..(a + l) as usize]).unwrap()).into_any()
            }
        }
    }
}
