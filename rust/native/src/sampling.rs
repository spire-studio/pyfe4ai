//! The Gaussian samplers of pyfe4ai.primitives.sampling: the rounded sampler
//! on given bytes (gauss), FACCT (exact, double sampling) for narrow and wide
//! widths, the rounded sampler of wide widths (gauss_bigint), and the noise
//! argument of the encryption calls (FacctArg, with_noise!).

use crate::drbg::{os_random, CallDrbg};
use crate::values::{py_index, u256_arg};
use pyfe4ai_core::bigauss;
use pyfe4ai_core::facct;
use pyfe4ai_core::facct_wide;
use pyfe4ai_core::gauss::{gauss_bytes, gauss_fill};
use pyfe4ai_core::rlwe;
use pyfe4ai_core::{parallel_for, Word, U256};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::sync::Mutex;

#[pyfunction]
pub(crate) fn gauss(py: Python<'_>, rnd: &Bound<'_, PyAny>, count: usize, sigma: f64) -> PyResult<Vec<i64>> {
    let need = gauss_bytes(count);
    let buf = if rnd.is_none() { os_random(py, need)? } else { rnd.extract::<Vec<u8>>()? };
    if buf.len() < need {
        return Err(PyValueError::new_err("not enough random bytes"));
    }
    let mut out = vec![0i64; count];
    gauss_fill(&buf, sigma, &mut out);
    Ok(out)
}

/// The tables of discrete_gaussian_facct for one width (see pyfe4ai-core facct)
/// and `exact(x, y)`, the Python acceptance threshold for the rare undecided tests.
#[pyclass(frozen, module = "pyfe4ai._rust_native", name = "FacctTable")]
pub(crate) struct FacctObj {
    pub(crate) table: facct::Table,
    exact: Py<PyAny>,
}

#[pymethods]
impl FacctObj {
    /// The tables; a few fixed-point thresholds are checked against `exact`
    /// (ValueError if c is not precise enough for the 2^-100 band).
    #[new]
    fn new(k: u64, y_bits: u32, cdt: Vec<Bound<'_, PyAny>>, c_mant: u128, c_exp: i32, exact: Py<PyAny>) -> PyResult<Self> {
        if cdt.len() != facct::CDT_SIZE || k == 0 || k > 1 << 50 {
            return Err(PyValueError::new_err("unsupported FACCT tables"));
        }
        let mut t = [U256::ZERO; facct::CDT_SIZE];
        for (dst, v) in t.iter_mut().zip(cdt.iter()) {
            *dst = u256_arg(v)?;
        }
        let obj = FacctObj { table: facct::Table { k, y_bits, cdt: t, c_mant, c_exp }, exact };
        let err = Mutex::new(None);
        let consistent = facct::check_table(&obj.table, &exact_fn(&obj, &err));
        exact_err(&err)?;
        if !consistent {
            return Err(PyValueError::new_err("FACCT tables: c disagrees with the exact acceptance thresholds"));
        }
        Ok(obj)
    }
}

/// `exact` as a closure for the kernels (with the GIL); a Python error is kept in `err`.
pub(crate) fn exact_fn<'a>(obj: &'a FacctObj, err: &'a Mutex<Option<PyErr>>) -> impl Fn(u64, u64) -> U256 + Sync + 'a {
    move |x, y| {
        Python::attach(|py| match obj.exact.call1(py, (x, y)).and_then(|v| u256_arg(v.bind(py))) {
            Ok(v) => v,
            Err(e) => {
                err.lock().unwrap().get_or_insert(e);
                U256::ZERO
            }
        })
    }
}

pub(crate) fn exact_err(err: &Mutex<Option<PyErr>>) -> PyResult<()> {
    match err.lock().unwrap().take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// discrete_gaussian_facct(sigma, count, rng) for the width of `table`: rng is
/// XofRandom(xof_seed) if given, else the call's DRBG (stream 0, or `seed`).
#[pyfunction]
#[pyo3(signature = (table, count, xof_seed=None, seed=None))]
pub(crate) fn facct_sample(py: Python<'_>, table: &Bound<'_, FacctObj>, count: usize, xof_seed: Option<Vec<u8>>,
                seed: Option<Vec<u8>>) -> PyResult<Py<PyList>> {
    let obj = table.get();
    let err = Mutex::new(None);
    let exact = exact_fn(obj, &err);
    let mut out = vec![0i64; count];
    match xof_seed {
        Some(s) => {
            let mut bits = facct::XofBits::new(&s);
            py.detach(|| facct::sample(&obj.table, &mut out, &mut bits, &exact));
        }
        None => {
            let drbg = CallDrbg::new(seed.as_deref())?;
            let mut offset = 0u64;
            let mut bits = facct::ByteBits::new(4096, |buf: &mut [u8]| {
                drbg.fill_at(0, offset, buf);
                offset += buf.len() as u64;
                buf.len()
            });
            py.detach(|| facct::sample(&obj.table, &mut out, &mut bits, &exact));
        }
    }
    exact_err(&err)?;
    Ok(PyList::new(py, out)?.unbind())
}

/// int.from_bytes(b, "little") (a new reference, or null with an exception set).
unsafe fn long_from_le(b: &[u8]) -> *mut pyo3::ffi::PyObject {
    #[cfg(Py_3_13)]
    {
        // Py_ASNATIVEBYTES_LITTLE_ENDIAN
        pyo3::ffi::PyLong_FromUnsignedNativeBytes(b.as_ptr().cast(), b.len(), 1)
    }
    #[cfg(not(Py_3_13))]
    {
        pyo3::ffi::_PyLong_FromByteArray(b.as_ptr(), b.len(), 1, 0)
    }
}

/// A non-negative little-endian magnitude as a Python int, negated if `neg`.
fn mag_pyint<'py>(py: Python<'py>, mag: &[u64], neg: bool) -> PyResult<Bound<'py, PyAny>> {
    let mut b = [0u8; 8 * bigauss::MAG];
    for (i, w) in mag.iter().enumerate() {
        b[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
    }
    let v = unsafe { Bound::from_owned_ptr_or_err(py, long_from_le(&b[..8 * mag.len()])) };
    rlwe::wipe_bytes(&mut b);
    let v = v?;
    if neg { v.neg() } else { Ok(v) }
}

/// Most samples one gauss_bigint call takes (pyfe4ai.backend.MAX_NATIVE_BIGINT_COUNT).
const MAX_BIGINT_COUNT: usize = 1 << 24;

/// One chunk of gauss_bigint: its magnitudes, flags and the bytes of its undecided samples.
struct BigChunk {
    mags: Vec<u64>,
    flags: Vec<u8>,
    holes: Vec<(usize, Vec<u8>)>,
}

/// discrete_gaussian_bigint(sigma, count, rng) with the rounded sampler (see
/// pyfe4ai-core bigauss) at precision `prec` for sig = sig_mant 2^sig_exp
/// (sig_mant little-endian). The uniforms are getrandbits(prec) of an
/// XofRandom (`xof_head` = its len(seed) || seed) from call `xof_counter` on --
/// calls the caller has claimed (XofRandom._claim) -- if given, else of `rnd`
/// (bytes, tests), else of the call's DRBG (stream c for samples 256 c to
/// 256 c + 255). Returns (values,
/// holes): values[i] is the sample, or None when undecided, and holes the (i,
/// b1 bytes, b2 bytes) of those, for the reference formula. `widen` enlarges
/// the margin (tests).
#[pyfunction]
#[pyo3(signature = (count, prec, sig_mant, sig_exp, xof_head=None, xof_counter=0, rnd=None, seed=None, threads=1, widen=0))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gauss_bigint(py: Python<'_>, count: usize, prec: u32, sig_mant: &[u8], sig_exp: i64, xof_head: Option<Vec<u8>>,
                xof_counter: u64, rnd: Option<Vec<u8>>, seed: Option<Vec<u8>>, threads: usize, widen: u32)
                -> PyResult<(Py<PyList>, Py<PyList>)> {
    const CHUNK: usize = 256;
    if count > MAX_BIGINT_COUNT {
        return Err(PyValueError::new_err("too many samples for one native call"));
    }
    let p = bigauss::Params::new(prec, sig_mant, sig_exp, widen).map_err(PyValueError::new_err)?;
    let nb = p.nbytes();
    let per = 2 * nb;
    if let Some(b) = &rnd {
        if b.len() < per * count {
            return Err(PyValueError::new_err("not enough random bytes"));
        }
    }
    if xof_head.is_some() && xof_counter.checked_add(2 * count as u64).is_none() {
        return Err(pyo3::exceptions::PyOverflowError::new_err("XofRandom counter beyond 64 bits"));
    }
    let xof_head = xof_head.map(zeroize::Zeroizing::new); // the seed copy is wiped at the end
    let drbg = if xof_head.is_none() && rnd.is_none() { Some(CallDrbg::new(seed.as_deref())?) } else { None };
    let n_chunks = count.div_ceil(CHUNK);
    let mw = p.mag_words();
    let slots: Vec<Mutex<Option<BigChunk>>> = (0..n_chunks).map(|_| Mutex::new(None)).collect();
    py.detach(|| {
        parallel_for(n_chunks, threads, |c| {
            let (s, e) = (c * CHUNK, ((c + 1) * CHUNK).min(count));
            let m = e - s;
            let mut own = Vec::new();
            let bytes: &[u8] = match (&rnd, &xof_head, &drbg) {
                (Some(b), _, _) => &b[s * per..e * per],
                (None, Some(head), _) => {
                    own = vec![0u8; m * per];
                    for i in 0..m {
                        let ctr = xof_counter + 2 * (s + i) as u64;
                        bigauss::xof_call(head, ctr, &mut own[i * per..i * per + nb]);
                        bigauss::xof_call(head, ctr + 1, &mut own[i * per + nb..(i + 1) * per]);
                    }
                    &own
                }
                (None, None, Some(d)) => {
                    own = vec![0u8; m * per];
                    d.fill(c as u64, &mut own);
                    &own
                }
                _ => unreachable!(),
            };
            let mut out = BigChunk { mags: vec![0u64; mw * m], flags: vec![0u8; m], holes: Vec::new() };
            bigauss::fill(&p, bytes, &mut out.mags, &mut out.flags);
            for (i, &f) in out.flags.iter().enumerate() {
                if f == bigauss::UNDECIDED {
                    out.holes.push((s + i, bytes[i * per..(i + 1) * per].to_vec()));
                }
            }
            rlwe::wipe_bytes(&mut own);
            *slots[c].lock().unwrap() = Some(out);
        })
    });
    let values = PyList::empty(py);
    let holes = PyList::empty(py);
    for slot in slots {
        let Some(mut ch) = slot.into_inner().unwrap() else { continue };
        for (i, &f) in ch.flags.iter().enumerate() {
            if f == bigauss::UNDECIDED {
                values.append(py.None())?;
            } else {
                let mag = &ch.mags[mw * i..mw * (i + 1)];
                values.append(mag_pyint(py, mag, f == bigauss::NEGATIVE)?)?;
            }
        }
        for (i, b) in ch.holes.iter_mut() {
            holes.append((*i, PyBytes::new(py, &b[..nb]), PyBytes::new(py, &b[nb..])))?;
            rlwe::wipe_bytes(b);
        }
        for w in ch.mags.iter_mut() {
            unsafe { std::ptr::write_volatile(w, 0) };
        }
        rlwe::wipe_bytes(&mut ch.flags); // the signs
    }
    Ok((values.unbind(), holes.unbind()))
}

/// The native approximation of X = sig sqrt(-2 ln u1) cos(2 pi u2) for one pair of
/// uniforms (bytes of getrandbits(prec)), for the tests: (P, sh, negative, words)
/// with |X| ~ P 2^-sh, or None (u1 = 1, or the square root check failed).
#[pyfunction]
pub(crate) fn gauss_bigint_approx(py: Python<'_>, prec: u32, sig_mant: &[u8], sig_exp: i64, b1: &[u8], b2: &[u8])
                       -> PyResult<Py<PyAny>> {
    let p = bigauss::Params::new(prec, sig_mant, sig_exp, 0).map_err(PyValueError::new_err)?;
    if b1.len() != p.nbytes() || b2.len() != p.nbytes() {
        return Err(PyValueError::new_err("each uniform has ceil(prec / 8) bytes"));
    }
    match bigauss::approx(&p, b1, b2) {
        None => Ok(py.None()),
        Some((prod, sh, neg)) => {
            let b: Vec<u8> = prod.iter().flat_map(|w| w.to_le_bytes()).collect();
            let v = unsafe { Bound::from_owned_ptr_or_err(py, long_from_le(&b))? };
            Ok((v, sh, neg, p.words()).into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// sampling.discrete_gaussian_facct(sigma, count, rng) for the width of
/// `table`, returning (samples, used):
/// * rng an XofRandom (`xof_head` = its `_seed`, from call `xof_counter`): run
///   with the GIL held -- the number of calls depends on the samples, so the
///   caller holds the XofRandom claim lock -- and `used` is the counter after
///   the last call;
/// * `rnd` (tests): getrandbits over these bytes, as `_OSRandomBuffer` reads the
///   OS stream; `used` is the number of bytes read;
/// * else the call's DRBG, samples 256 c to 256 c + 255 from stream c (on
///   `threads` threads); `used` is 0.
#[pyfunction]
#[pyo3(signature = (table, count, xof_head=None, xof_counter=0, rnd=None, seed=None, threads=1))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn facct_draw(py: Python<'_>, table: &Bound<'_, FacctObj>, count: usize, xof_head: Option<Vec<u8>>, xof_counter: u64,
              rnd: Option<Vec<u8>>, seed: Option<Vec<u8>>, threads: usize) -> PyResult<(Py<PyList>, u128)> {
    const CHUNK: usize = 256;
    if count > MAX_BIGINT_COUNT {
        return Err(PyValueError::new_err("too many samples for one native call"));
    }
    let obj = table.get();
    let err = Mutex::new(None);
    let exact = exact_fn(obj, &err);
    let mut out = vec![0i64; count];
    let used = if let Some(head) = xof_head {
        let head = zeroize::Zeroizing::new(head);
        let mut bits = facct::XofBits::at(&head, xof_counter);
        facct::sample(&obj.table, &mut out, &mut bits, &exact); // the GIL stays held
        if bits.overflowed() {
            exact_err(&err)?;
            return Err(pyo3::exceptions::PyOverflowError::new_err("XofRandom counter beyond 64 bits"));
        }
        bits.next_call()
    } else if let Some(data) = rnd {
        let data = zeroize::Zeroizing::new(data);
        let mut at = 0usize;
        let mut bits = facct::ByteBits::new(256, |buf: &mut [u8]| {
            let n = buf.len().min(data.len() - at);
            buf[..n].copy_from_slice(&data[at..at + n]);
            at += n;
            n
        });
        py.detach(|| facct::sample(&obj.table, &mut out, &mut bits, &exact));
        if bits.failed {
            return Err(PyValueError::new_err("not enough random bytes"));
        }
        bits.consumed() as u128
    } else {
        let drbg = CallDrbg::new(seed.as_deref())?;
        let slots: Vec<Mutex<&mut [i64]>> = out.chunks_mut(CHUNK).map(Mutex::new).collect();
        py.detach(|| {
            parallel_for(slots.len(), threads, |c| {
                let mut offset = 0u64;
                let mut bits = facct::ByteBits::new(4096, |buf: &mut [u8]| {
                    drbg.fill_at(c as u64, offset, buf);
                    offset += buf.len() as u64;
                    buf.len()
                });
                facct::sample(&obj.table, &mut slots[c].lock().unwrap(), &mut bits, &exact);
            })
        });
        0
    };
    exact_err(&err)?;
    let list = PyList::new(py, &out)?.unbind();
    for v in out.iter_mut() {
        unsafe { std::ptr::write_volatile(v, 0) };
    }
    Ok((list, used))
}

/// The tables of discrete_gaussian_facct for a wide width (2^50 < k < 2^448,
/// see pyfe4ai-core facct_wide) and `exact(x, y)`, the Python threshold.
#[pyclass(frozen, module = "pyfe4ai._rust_native", name = "FacctWideTable")]
pub(crate) struct FacctWideObj {
    table: facct_wide::WideTable,
    exact: Py<PyAny>,
}

#[pymethods]
impl FacctWideObj {
    /// The tables; a few fixed-point thresholds are checked against `exact`
    /// (ValueError if c is not precise enough for the 2^-100 band).
    #[new]
    fn new(k: &Bound<'_, PyAny>, y_bits: u32, cdt: Vec<Bound<'_, PyAny>>, c_mant: u128, c_exp: i32, exact: Py<PyAny>)
           -> PyResult<Self> {
        let py = k.py();
        let bad = || PyValueError::new_err("unsupported wide FACCT tables");
        let kb = py_index(k)?
            .call_method1(intern!(py, "to_bytes"), (8 * facct_wide::W, intern!(py, "little")))
            .map_err(|_| bad())?;
        let kb = kb.cast::<PyBytes>()?.as_bytes();
        let mut kw = [0u64; facct_wide::W];
        for (i, w) in kw.iter_mut().enumerate() {
            *w = u64::from_le_bytes(kb[8 * i..8 * i + 8].try_into().unwrap());
        }
        if cdt.len() != facct::CDT_SIZE || !facct_wide::valid(&kw, y_bits) {
            return Err(bad());
        }
        let mut t = [U256::ZERO; facct::CDT_SIZE];
        for (dst, v) in t.iter_mut().zip(cdt.iter()) {
            *dst = u256_arg(v)?;
        }
        let obj = FacctWideObj { table: facct_wide::WideTable { k: kw, y_bits, cdt: t, c_mant, c_exp }, exact };
        let err = Mutex::new(None);
        let consistent = facct_wide::check_table(&obj.table, &exact_wide_fn(&obj, &err));
        exact_err(&err)?;
        if !consistent {
            return Err(PyValueError::new_err("FACCT tables: c disagrees with the exact acceptance thresholds"));
        }
        Ok(obj)
    }
}

/// `exact` of a wide table as a closure for the kernel (with the GIL); a Python
/// error is kept in `err`.
fn exact_wide_fn<'a>(obj: &'a FacctWideObj, err: &'a Mutex<Option<PyErr>>) -> impl Fn(u64, &[u64]) -> U256 + Sync + 'a {
    move |x, y| {
        Python::attach(|py| {
            let mut b = [0u8; 8 * facct_wide::W];
            for (i, w) in y.iter().enumerate() {
                b[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
            }
            let yi = unsafe { Bound::from_owned_ptr_or_err(py, long_from_le(&b)) };
            rlwe::wipe_bytes(&mut b);
            match yi.and_then(|yi| obj.exact.call1(py, (x, yi))).and_then(|v| u256_arg(v.bind(py))) {
                Ok(v) => v,
                Err(e) => {
                    err.lock().unwrap().get_or_insert(e);
                    U256::ZERO
                }
            }
        })
    }
}

/// discrete_gaussian_facct for a wide table: as facct_draw, the samples as
/// Python ints.
#[pyfunction]
#[pyo3(signature = (table, count, xof_head=None, xof_counter=0, rnd=None, seed=None, threads=1))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn facct_draw_wide(py: Python<'_>, table: &Bound<'_, FacctWideObj>, count: usize, xof_head: Option<Vec<u8>>,
                   xof_counter: u64, rnd: Option<Vec<u8>>, seed: Option<Vec<u8>>, threads: usize)
                   -> PyResult<(Py<PyList>, u128)> {
    const CHUNK: usize = 256;
    const W: usize = facct_wide::W;
    if count > MAX_BIGINT_COUNT {
        return Err(PyValueError::new_err("too many samples for one native call"));
    }
    let obj = table.get();
    let err = Mutex::new(None);
    let exact = exact_wide_fn(obj, &err);
    let mut mags = vec![0u64; W * count];
    let mut neg = vec![0u8; count];
    let used = if let Some(head) = xof_head {
        let head = zeroize::Zeroizing::new(head);
        let mut bits = facct::XofBits::at(&head, xof_counter);
        facct_wide::sample(&obj.table, &mut mags, &mut neg, &mut bits, &exact); // the GIL stays held
        if bits.overflowed() {
            exact_err(&err)?;
            return Err(pyo3::exceptions::PyOverflowError::new_err("XofRandom counter beyond 64 bits"));
        }
        bits.next_call()
    } else if let Some(data) = rnd {
        let data = zeroize::Zeroizing::new(data);
        let mut at = 0usize;
        let mut bits = facct::ByteBits::new(256, |buf: &mut [u8]| {
            let n = buf.len().min(data.len() - at);
            buf[..n].copy_from_slice(&data[at..at + n]);
            at += n;
            n
        });
        py.detach(|| facct_wide::sample(&obj.table, &mut mags, &mut neg, &mut bits, &exact));
        if bits.failed {
            return Err(PyValueError::new_err("not enough random bytes"));
        }
        bits.consumed() as u128
    } else {
        let drbg = CallDrbg::new(seed.as_deref())?;
        let slots: Vec<Mutex<(&mut [u64], &mut [u8])>> =
            mags.chunks_mut(W * CHUNK).zip(neg.chunks_mut(CHUNK)).map(Mutex::new).collect();
        py.detach(|| {
            parallel_for(slots.len(), threads, |c| {
                let mut offset = 0u64;
                let mut bits = facct::ByteBits::new(4096, |buf: &mut [u8]| {
                    drbg.fill_at(c as u64, offset, buf);
                    offset += buf.len() as u64;
                    buf.len()
                });
                let mut g = slots[c].lock().unwrap();
                let (m, n) = &mut *g;
                facct_wide::sample(&obj.table, m, n, &mut bits, &exact);
            })
        });
        0
    };
    exact_err(&err)?;
    let list = PyList::empty(py);
    for i in 0..count {
        list.append(mag_pyint(py, &mags[W * i..W * (i + 1)], neg[i] == 1)?)?;
    }
    for w in mags.iter_mut() {
        unsafe { std::ptr::write_volatile(w, 0) };
    }
    rlwe::wipe_bytes(&mut neg);
    Ok((list.unbind(), used))
}

/// The noise of an encryption call: FACCT with the tables of (sigma2, sigma3), else rounded.
pub(crate) type FacctArg = Option<(Py<FacctObj>, Py<FacctObj>)>;

macro_rules! with_noise {
    ($py:expr, $facct:expr, $noise:ident, $err:ident, $body:block) => {{
        let $err = Mutex::new(None);
        let tables = $facct.as_ref().map(|(a, b)| (a.bind($py).get(), b.bind($py).get()));
        let (e2, e3) = match tables {
            Some((a, b)) => (Some(exact_fn(a, &$err)), Some(exact_fn(b, &$err))),
            None => (None, None),
        };
        let $noise = match (tables, e2.as_ref(), e3.as_ref()) {
            (Some((a, b)), Some(f2), Some(f3)) => Noise::Facct(FacctNoise {
                sigma2: &a.table, sigma3: &b.table, exact2: f2, exact3: f3,
            }),
            _ => Noise::Rounded,
        };
        let out = $body;
        drop($noise);
        exact_err(&$err)?;
        out
    }};
}
pub(crate) use with_noise;
