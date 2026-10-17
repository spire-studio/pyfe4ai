//! pyfe4ai._rust_native: the Rust backend of pyfe4ai.utils.native_backend.
//!
//! The lattice functions back the Ring-LWE and dense-LWE schemes (checked bit
//! for bit against their Python code by the tests); `ec` adds ristretto255 and
//! `pairing` BLS12-381. Values cross the
//! boundary as Python ints, decimal strings (the ciphertext format) or packed
//! little-endian words in bytes; the GIL is released around the kernels.

use pyfe4ai_core::codec::{bin_read, bin_size, bin_wb, bin_write};
use pyfe4ai_core::gauss::{gauss_bytes, gauss_fill};
use pyfe4ai_core::keccak::{hash_to_zq, shake256};
use pyfe4ai_core::facct;
use pyfe4ai_core::rlwe::{self, EncKey, FacctNoise, Noise, Precomputed, Randomness};
use pyfe4ai_core::{osrand, parallel_for, with_ring, AnyRing, Modulus, Ring, Word, U256};
use pyo3::intern;
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList, PyString, PyTuple};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::sync::{Arc, Mutex};

mod ec;
mod msm;
mod pairing;

// ------------------------------------------------------------------ DRBG

/// Randomness of one native call: ChaCha20 keyed by one 32-byte seed from the OS
/// (or injected by a test), with stream c for chunk c, so that no two chunks --
/// hence no two threads -- share a stream. Seed and generator states are wiped
/// when the call ends; nothing persists across calls.
struct CallDrbg {
    seed: zeroize::Zeroizing<[u8; 32]>,
}

impl CallDrbg {
    fn new(seed: Option<&[u8]>) -> PyResult<Self> {
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
    fn rng(&self, stream: u64) -> ChaCha20Rng {
        let mut rng = ChaCha20Rng::from_seed(*self.seed);
        rng.set_stream(stream);
        rng
    }

    /// Bytes of stream `stream` from its start.
    fn fill(&self, stream: u64, out: &mut [u8]) {
        self.fill_at(stream, 0, out);
    }

    /// Bytes of stream `stream` from byte `offset` (a multiple of 4).
    fn fill_at(&self, stream: u64, offset: u64, out: &mut [u8]) {
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
fn wipe_rng(rng: &mut ChaCha20Rng) {
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
fn drbg_bytes(py: Python<'_>, seed: &[u8], stream: u64, n: usize) -> PyResult<Py<PyBytes>> {
    let d = CallDrbg::new(Some(seed))?;
    let mut out = vec![0u8; n];
    d.fill(stream, &mut out);
    Ok(PyBytes::new(py, &out).unbind())
}

// ------------------------------------------------------------------ context

/// A ring Z_q[x] / (x^n + 1) with its NTT tables (ctx_new).
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
struct RingCtx {
    ring: Arc<AnyRing>,
    q: Py<PyAny>,
}

fn u256_arg(ob: &Bound<'_, PyAny>) -> PyResult<U256> {
    let v = py_index(ob)?;
    let err = || PyValueError::new_err("needs a non-negative 256-bit integer");
    if v.lt(0)? {
        return Err(err());
    }
    let b = v.call_method1(intern!(ob.py(), "to_bytes"), (32, intern!(ob.py(), "little"))).map_err(|_| err())?;
    Ok(U256::read_le(b.cast::<PyBytes>()?.as_bytes()))
}

fn py_index<'py>(ob: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
fn word_pyint<'py, W: Word>(py: Python<'py>, v: W) -> PyResult<Bound<'py, PyAny>> {
    if W::BITS <= 128 {
        return Ok(v.to_u128().into_pyobject(py)?.into_any());
    }
    let mut b = vec![0u8; (W::BITS / 8) as usize];
    v.write_le(&mut b);
    py.get_type::<PyInt>().call_method1(intern!(py, "from_bytes"), (PyBytes::new(py, &b), intern!(py, "little")))
}

/// int(str(ob)) % q (the semantics of canonical_ring_ciphertext).
fn str_int_mod<W: Word>(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<W> {
    let int = ob.py().import("builtins")?.getattr("int")?;
    pyint_word(&int.call1((ob.str()?,))?.rem(q)?)
}

/// ob % q for an exact int, without raising (None: not an exact int, a negative
/// value below -2^63, or a positive one Python must reduce).
fn int_mod_fast<W: Word>(ob: &Bound<'_, PyAny>, m: &Modulus<W>) -> Option<W> {
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
fn index_mod<W: Word>(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, m: &Modulus<W>) -> PyResult<W> {
    if let Some(x) = int_mod_fast(ob, m) {
        return Ok(x);
    }
    let v = py_index(ob)?;
    if let Some(x) = int_mod_fast(&v, m) {
        return Ok(x);
    }
    pyint_word(&v.rem(q)?)
}

/// Values of a sequence mod q: decimal strings by the fast path, other strings
/// as int(s), anything else by __index__ (str_semantics: int(str(v)) for all).
fn read_values<W: Word>(seq: &Bound<'_, PyAny>, n: Option<usize>, q: &Bound<'_, PyAny>, m: &Modulus<W>,
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
    if let Ok(list) = seq.cast::<PyList>() {
        check(list.len())?;
        return list.iter().map(|ob| one(&ob)).collect();
    }
    let items: Vec<Bound<'_, PyAny>> = seq.try_iter()?.collect::<PyResult<_>>()?;
    check(items.len())?;
    items.iter().map(one).collect()
}

/// Rows of values (each of n values, or any length).
fn read_rows<W: Word>(rows: &Bound<'_, PyAny>, n: Option<usize>, q: &Bound<'_, PyAny>, m: &Modulus<W>)
    -> PyResult<Vec<Vec<W>>> {
    rows.try_iter()?.map(|row| read_values(&row?, n, q, m, false)).collect()
}

fn get_ring(ctx: &Bound<'_, RingCtx>) -> Arc<AnyRing> {
    ctx.get().ring.clone()
}

#[pyfunction]
fn ctx_new(py: Python<'_>, q: &Bound<'_, PyAny>, n: usize, root: &Bound<'_, PyAny>) -> PyResult<RingCtx> {
    let (qv, rv) = (u256_arg(q)?, u256_arg(root)?);
    let ring = py.detach(|| AnyRing::new(qv, n, rv)).map_err(PyValueError::new_err)?;
    Ok(RingCtx { ring: Arc::new(ring), q: py_index(q)?.unbind() })
}

#[pyfunction]
fn ctx_info(py: Python<'_>, ctx: &Bound<'_, RingCtx>) -> PyResult<Py<PyAny>> {
    let r = &ctx.get().ring;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("word_bits", r.word_bits())?;
    d.set_item("n", r.n())?;
    d.set_item("q_bits", r.qbits())?;
    Ok(d.into_any().unbind())
}

// Words cross the boundary as packed native-endian (little-endian) bytes.
fn words_to_bytes<W: Word>(py: Python<'_>, w: &[W]) -> Py<PyBytes> {
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

fn bytes_to_words<W: Word>(b: &[u8], count: usize) -> PyResult<Vec<W>> {
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

fn dec_list<'py, W: Word>(py: Python<'py>, vals: &[W]) -> PyResult<Bound<'py, PyList>> {
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
struct DecBuf {
    text: Vec<u8>,
    off: Vec<u32>,
}

impl DecBuf {
    fn new<W: Word>(vals: &[W]) -> Self {
        let mut text = Vec::with_capacity(vals.len() * (W::BITS as usize * 3 / 10 + 2));
        let mut off = Vec::with_capacity(vals.len() + 1);
        for v in vals {
            off.push(text.len() as u32);
            v.format_dec(&mut text);
        }
        off.push(text.len() as u32);
        DecBuf { text, off }
    }
    fn list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items = self
            .off
            .windows(2)
            .map(|w| ascii_str(py, &self.text[w[0] as usize..w[1] as usize]))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }
}

#[pyfunction]
fn pack(py: Python<'_>, ctx: &Bound<'_, RingCtx>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let q = ctx.get().q.bind(py);
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => Ok(words_to_bytes(py, &read_values(values, None, q, &r.m, false)?)))
}

#[pyfunction]
#[pyo3(signature = (ctx, data, as_digits=false))]
fn unpack(py: Python<'_>, ctx: &Bound<'_, RingCtx>, data: &[u8], as_digits: bool) -> PyResult<Py<PyList>> {
    let ring = get_ring(ctx);
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
fn ntt(py: Python<'_>, ctx: &Bound<'_, RingCtx>, data: &[u8], inverse: bool) -> PyResult<Py<PyBytes>> {
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let x = bytes_to_words(data, r.n)?;
        let mut y = x.clone();
        if inverse { r.inverse(&x, &mut y) } else { r.forward(&x, &mut y) }
        Ok(words_to_bytes(py, &y))
    })
}

#[pyfunction]
fn ntt_mont(py: Python<'_>, ctx: &Bound<'_, RingCtx>, data: &[u8]) -> PyResult<Py<PyBytes>> {
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let x = bytes_to_words(data, r.n)?;
        let mut y = x.clone();
        r.forward_mont(&x, &mut y);
        Ok(words_to_bytes(py, &y))
    })
}

#[pyfunction]
fn poly_mul(py: Python<'_>, ctx: &Bound<'_, RingCtx>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let (x, y) = (bytes_to_words(a, r.n)?, bytes_to_words(b, r.n)?);
        Ok(words_to_bytes(py, &r.poly_mul(&x, &y)))
    })
}

/// sum_i coeffs[i] rows[i] mod q as decimal strings.
#[pyfunction]
fn lincomb(py: Python<'_>, ctx: &Bound<'_, RingCtx>, rows: &Bound<'_, PyAny>, coeffs: &Bound<'_, PyAny>)
    -> PyResult<Py<PyList>> {
    let q = ctx.get().q.bind(py);
    let ring = get_ring(ctx);
    let rows: Vec<Vec<u8>> = rows.try_iter()?.map(|r| r?.extract::<Vec<u8>>()).collect::<PyResult<_>>()?;
    with_ring!(&*ring, r => {
        let ys = read_values(coeffs, None, q, &r.m, false)?;
        if rows.len() != ys.len() {
            return Err(PyValueError::new_err("one coefficient per row"));
        }
        let mut acc = vec![Word::ZERO; r.n];
        for (row, &y) in rows.iter().zip(ys.iter()) {
            let v = bytes_to_words(row, r.n)?;
            let ym = r.m.to_mont(y);
            for i in 0..r.n {
                acc[i] = r.m.add(acc[i], r.m.mul(v[i], ym));
            }
        }
        Ok(dec_list(py, &acc)?.unbind())
    })
}

// ------------------------------------------------------------------ kernels

fn os_random(py: Python<'_>, n: usize) -> PyResult<Vec<u8>> {
    let mut b = vec![0u8; n];
    py.detach(|| osrand::fill(&mut b)).map_err(|e| PyOSError::new_err(e.to_string()))?;
    Ok(b)
}

#[pyfunction]
fn gauss(py: Python<'_>, rnd: &Bound<'_, PyAny>, count: usize, sigma: f64) -> PyResult<Vec<i64>> {
    let need = gauss_bytes(count);
    let buf = if rnd.is_none() { os_random(py, need)? } else { rnd.extract::<Vec<u8>>()? };
    if buf.len() < need {
        return Err(PyValueError::new_err("not enough random bytes"));
    }
    let mut out = vec![0i64; count];
    gauss_fill(&buf, sigma, &mut out);
    Ok(out)
}

#[pyfunction(name = "shake256")]
fn shake256_py(py: Python<'_>, data: &[u8], length: usize) -> PyResult<Py<PyBytes>> {
    let mut out = vec![0u8; length];
    py.detach(|| shake256(data, &mut out));
    Ok(PyBytes::new(py, &out).unbind())
}

#[pyfunction(name = "hash_to_zq")]
fn hash_to_zq_py(py: Python<'_>, ctx: &Bound<'_, RingCtx>, material: &[u8], count: usize) -> PyResult<Py<PyBytes>> {
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let mut out = vec![Word::ZERO; count];
        py.detach(|| hash_to_zq(&r.m, material, &mut out));
        Ok(words_to_bytes(py, &out))
    })
}

#[pyfunction]
fn random_bytes(py: Python<'_>, n: usize) -> PyResult<Py<PyBytes>> {
    Ok(PyBytes::new(py, &os_random(py, n)?).unbind())
}

// --------------------------------------------------------- encryption key

enum AnyKey {
    W64(EncKey<u64>),
    W128(EncKey<u128>),
    W256(EncKey<U256>),
}

/// A slot's encryption key: NTT(a), NTT(pk_row) in Montgomery form, u, Delta.
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
struct EncKeyObj {
    ring: Arc<AnyRing>,
    q: Py<PyAny>,
    key: AnyKey,
}

macro_rules! with_key {
    ($obj:expr, $r:ident, $k:ident => $body:expr) => {
        match (&*$obj.ring, &$obj.key) {
            (AnyRing::W64($r), AnyKey::W64($k)) => $body,
            (AnyRing::W128($r), AnyKey::W128($k)) => $body,
            (AnyRing::W256($r), AnyKey::W256($k)) => $body,
            _ => unreachable!(),
        }
    };
}

#[pyfunction]
fn enc_key(py: Python<'_>, ctx: &Bound<'_, RingCtx>, a: &Bound<'_, PyAny>, pk: &Bound<'_, PyAny>,
           u: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>) -> PyResult<EncKeyObj> {
    let q = ctx.get().q.bind(py);
    let ring = get_ring(ctx);
    let n = ring.n();
    macro_rules! build {
        ($r:expr, $variant:ident) => {{
            let m = &$r.m;
            let av = read_values(a, Some(n), q, m, false)?;
            let pk_rows = read_rows(pk, Some(n), q, m)?;
            if pk_rows.is_empty() {
                return Err(PyValueError::new_err("pk needs at least one row"));
            }
            let u_rows = if u.is_none() {
                None
            } else {
                let rows = read_rows(u, Some(n), q, m)?;
                if rows.len() != pk_rows.len() {
                    return Err(PyValueError::new_err("one pad row per pk row"));
                }
                Some(rows)
            };
            let d = index_mod(delta, q, m)?;
            AnyKey::$variant(EncKey::new($r, &av, &pk_rows, u_rows.as_deref(), d))
        }};
    }
    let key = match &*ring {
        AnyRing::W64(r) => build!(r, W64),
        AnyRing::W128(r) => build!(r, W128),
        AnyRing::W256(r) => build!(r, W256),
    };
    Ok(EncKeyObj { ring, q: q.clone().unbind(), key })
}

/// Given bytes (tests), else the call's DRBG streams.
fn randomness<'a>(rnd: &'a Option<Vec<u8>>, stream: &'a (dyn Fn(usize, u64, &mut [u8]) + Sync)) -> Randomness<'a> {
    match rnd {
        Some(b) => Randomness::Given(b),
        None => Randomness::Stream(stream),
    }
}

fn core_err(e: rlwe::Error) -> PyErr {
    match e {
        rlwe::Error::Input(m) => PyValueError::new_err(m),
        rlwe::Error::Random(e) => PyOSError::new_err(e.to_string()),
    }
}

// ------------------------------------------------------------------ FACCT

/// The tables of discrete_gaussian_facct for one width (see pyfe4ai-core facct)
/// and `exact(x, y)`, the Python acceptance threshold for the rare undecided tests.
#[pyclass(frozen, module = "pyfe4ai._rust_native", name = "FacctTable")]
struct FacctObj {
    table: facct::Table,
    exact: Py<PyAny>,
}

#[pymethods]
impl FacctObj {
    #[new]
    fn new(k: u64, y_bits: u32, cdt: Vec<Bound<'_, PyAny>>, c_mant: u128, c_exp: i32, exact: Py<PyAny>) -> PyResult<Self> {
        if cdt.len() != facct::CDT_SIZE || k == 0 || k > 1 << 50 {
            return Err(PyValueError::new_err("unsupported FACCT tables"));
        }
        let mut t = [U256::ZERO; facct::CDT_SIZE];
        for (dst, v) in t.iter_mut().zip(cdt.iter()) {
            *dst = u256_arg(v)?;
        }
        Ok(FacctObj { table: facct::Table { k, y_bits, cdt: t, c_mant, c_exp }, exact })
    }
}

/// `exact` as a closure for the kernels (with the GIL); a Python error is kept in `err`.
fn exact_fn<'a>(obj: &'a FacctObj, err: &'a Mutex<Option<PyErr>>) -> impl Fn(u64, u64) -> U256 + Sync + 'a {
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

fn exact_err(err: &Mutex<Option<PyErr>>) -> PyResult<()> {
    match err.lock().unwrap().take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// discrete_gaussian_facct(sigma, count, rng) for the width of `table`: rng is
/// XofRandom(xof_seed) if given, else the call's DRBG (stream 0, or `seed`).
#[pyfunction]
#[pyo3(signature = (table, count, xof_seed=None, seed=None))]
fn facct_sample(py: Python<'_>, table: &Bound<'_, FacctObj>, count: usize, xof_seed: Option<Vec<u8>>,
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

/// The noise of an encryption call: FACCT with the tables of (sigma2, sigma3), else rounded.
type FacctArg = Option<(Py<FacctObj>, Py<FacctObj>)>;

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

/// MIFERingLWE._encrypt_rows: (ct0, ct1) as decimal strings, or format-1 bytes.
#[pyfunction]
#[pyo3(signature = (key, w_rows, sigma2, sigma3, rnd, binary=false, seed=None, facct=None))]
#[allow(clippy::too_many_arguments)]
fn encrypt(py: Python<'_>, key: &Bound<'_, EncKeyObj>, w_rows: &Bound<'_, PyAny>, sigma2: f64, sigma3: f64,
           rnd: &Bound<'_, PyAny>, binary: bool, seed: Option<Vec<u8>>, facct: FacctArg) -> PyResult<Py<PyAny>> {
    let obj = key.get();
    let ring = obj.ring.clone();
    let n = ring.n();
    let qob = obj.q.bind(py);
    with_key!(obj, r, k => {
        let rows = read_rows(w_rows, Some(n), qob, &r.m)?;
        let eta = k.eta;
        if rows.len() != eta {
            return Err(PyValueError::new_err("one plaintext row per pk row"));
        }
        let w: Vec<_> = rows.concat();
        let (mut c0, mut c1) = (vec![Word::ZERO; eta * n], vec![Word::ZERO; n]);
        if facct.is_some() {
            let given: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
            let drbg = CallDrbg::new(seed.as_deref())?;
            let (mut pos, mut offset) = (0usize, 0u64);
            let mut bits = facct::ByteBits::new(4096, |buf: &mut [u8]| match &given {
                Some(b) => {
                    let c = buf.len().min(b.len() - pos);
                    buf[..c].copy_from_slice(&b[pos..pos + c]);
                    pos += c;
                    c
                }
                None => {
                    drbg.fill_at(0, offset, buf);
                    offset += buf.len() as u64;
                    buf.len()
                }
            });
            with_noise!(py, facct, noise, err, {
                if let Noise::Facct(nz) = &noise {
                    py.detach(|| rlwe::encrypt_chunk_facct(r, k, &w, nz, &mut bits, &mut c0, &mut c1));
                }
            });
            if bits.failed {
                return Err(PyValueError::new_err("not enough random bytes"));
            }
        } else {
            let need = gauss_bytes(n) * (2 + eta);
            let mut rb = if rnd.is_none() {
                let mut b = vec![0u8; need];
                CallDrbg::new(seed.as_deref())?.fill(0, &mut b);
                b
            } else {
                rnd.extract::<Vec<u8>>()?
            };
            if rb.len() < need {
                return Err(PyValueError::new_err("not enough random bytes"));
            }
            py.detach(|| rlwe::encrypt_chunk(r, k, &w, sigma2, sigma3, &rb, &mut c0, &mut c1));
            rlwe::wipe_bytes(&mut rb);
        }
        if binary {
            let mut out = Vec::new();
            bin_write(ring.qbits(), r.logn, eta, &c0, &c1, &mut out);
            return Ok(PyBytes::new(py, &out).into_any().unbind());
        }
        let rows_out = PyList::new(py, c0.chunks(n).map(|row| dec_list(py, row)).collect::<PyResult<Vec<_>>>()?)?;
        Ok(PyTuple::new(py, [rows_out.into_any(), dec_list(py, &c1)?.into_any()])?.into_any().unbind())
    })
}

/// Per-chunk outputs of the tensor encryptions, formatted in the workers.
enum ChunkOut {
    Dec(DecBuf, DecBuf),
    Bin(Vec<u8>),
}

fn chunk_out<W: Word>(r: &Ring<W>, binary: bool, c0: &[W], c1: &[W]) -> ChunkOut {
    if binary {
        let mut out = Vec::with_capacity(bin_size(1, r.n, bin_wb(r.m.bits)));
        bin_write(r.m.bits, r.logn, 1, c0, c1, &mut out);
        ChunkOut::Bin(out)
    } else {
        ChunkOut::Dec(DecBuf::new(c0), DecBuf::new(c1))
    }
}

fn chunk_list(py: Python<'_>, outs: Vec<Option<ChunkOut>>, n: usize, len: usize) -> PyResult<Py<PyList>> {
    let items = outs
        .into_iter()
        .enumerate()
        .map(|(j, o)| -> PyResult<Bound<'_, PyAny>> {
            let k = n.min(len - j * n);
            Ok(match o.expect("every chunk is encrypted") {
                ChunkOut::Bin(b) => PyTuple::new(py, [PyBytes::new(py, &b).into_any(), k.into_pyobject(py)?.into_any()])?.into_any(),
                ChunkOut::Dec(a, b) => PyTuple::new(py, [
                    PyList::new(py, [a.list(py)?])?.into_any(), b.list(py)?.into_any(), k.into_pyobject(py)?.into_any(),
                ])?.into_any(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.unbind())
}

fn materials_of(mats: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Vec<u8>>>> {
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

/// MCFE ndarray encryption of one layer (eta = 1); see pyfe4ai-core rlwe::encrypt_tensor.
#[pyfunction]
#[pyo3(signature = (key, x, pad_materials, sigma2, sigma3, rnd, threads, binary=false, seed=None, facct=None))]
#[allow(clippy::too_many_arguments)]
fn encrypt_tensor(py: Python<'_>, key: &Bound<'_, EncKeyObj>, x: &Bound<'_, PyAny>, pad_materials: &Bound<'_, PyAny>,
                  sigma2: f64, sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, binary: bool,
                  seed: Option<Vec<u8>>, facct: FacctArg) -> PyResult<Py<PyList>> {
    let obj = key.get();
    let xs = int64_buffer(x)?;
    let mats = materials_of(pad_materials)?;
    let rb: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    let stream = |c: usize, off: u64, buf: &mut [u8]| drbg.fill_at(c as u64, off, buf);
    with_key!(obj, r, k => {
        let n = r.n;
        let chunks = if xs.is_empty() { 0 } else { (xs.len() + n - 1) / n };
        let outs: Vec<Mutex<Option<ChunkOut>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
        let refs: Option<Vec<&[u8]>> = mats.as_ref().map(|m| m.iter().map(|v| v.as_slice()).collect());
        with_noise!(py, facct, noise, err, {
            py.detach(|| rlwe::encrypt_tensor(r, k, &xs, refs.as_deref(), sigma2, sigma3, &noise, randomness(&rb, &stream),
                threads, |j, c0, c1| *outs[j].lock().unwrap() = Some(chunk_out(r, binary, c0, c1))))
                .map_err(core_err)?
        });
        chunk_list(py, outs.into_iter().map(|m| m.into_inner().unwrap()).collect(), n, xs.len())
    })
}

/// An int64 buffer (numpy array, bytes, ...) as a vector.
fn int64_buffer(x: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
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

// --------------------------------------------------------------- decryption

/// One gathered ciphertext value list: plain decimal strings copied, others for Python.
struct GText {
    text: Vec<u8>,
    pos: Vec<(u32, u32)>,
}

struct Gathered<'py> {
    t: GText,
    slow: Vec<(usize, Bound<'py, PyAny>)>,
}

const SLOW: u32 = u32::MAX;

/// The values of a list or tuple of n items; ASCII strings of at most max_len
/// characters are copied for the parser (the rest go to Python).
fn gather<'py>(seq: &Bound<'py, PyAny>, n: usize, max_len: usize) -> Option<Gathered<'py>> {
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
    fn object(&self, py: Python<'py>, i: usize) -> Bound<'py, PyAny> {
        match self.slow.iter().find(|(k, _)| *k == i) {
            Some((_, o)) => o.clone().into_any(),
            None => {
                let (a, l) = self.t.pos[i];
                PyString::new(py, std::str::from_utf8(&self.t.text[a as usize..(a + l) as usize]).unwrap()).into_any()
            }
        }
    }
}

enum Slot<'py> {
    Dec(Vec<Gathered<'py>>), // eta rows of ct0, then ct1
    Bin(Bound<'py, PyBytes>),
}

/// A slot's values: parsed decimal texts, or a binary ciphertext decoded later without the GIL.
enum SlotVals<'a, W> {
    Words(Vec<W>),
    Bin(&'a [u8]),
}

/// MIFERingLWE._decrypt_with_pad_term for every chunk of a layer: returns
/// (int64 bytes, None mask bytes | None).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decrypt_tensor(py: Python<'_>, ctx: &Bound<'_, RingCtx>, chunks: &Bound<'_, PyAny>, zs: &Bound<'_, PyAny>,
                  ks: Vec<i64>, sk_hats: Vec<Vec<u8>>, ys: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>,
                  bound: &Bound<'_, PyAny>, threads: usize) -> PyResult<Py<PyAny>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => decrypt_tensor_w(py, r, &q, chunks, zs, ks, sk_hats, ys, delta, bound, threads))
}

#[allow(clippy::too_many_arguments)]
fn decrypt_tensor_w<W: Word>(py: Python<'_>, r: &Ring<W>, q: &Bound<'_, PyAny>, chunks: &Bound<'_, PyAny>,
                             zs: &Bound<'_, PyAny>, ks: Vec<i64>, sk_hats: Vec<Vec<u8>>, ys: &Bound<'_, PyAny>,
                             delta: &Bound<'_, PyAny>, bound: &Bound<'_, PyAny>, threads: usize)
    -> PyResult<Py<PyAny>> {
    let m = &r.m;
    let n = r.n;
    let max_len = W::DEC_DIGITS + 1;
    let fallback = || PyNotImplementedError::new_err("input layout not handled natively");
    let chunk_items: Vec<Bound<'_, PyAny>> = chunks.try_iter()?.collect::<PyResult<_>>()?;
    let z_items: Vec<Bound<'_, PyAny>> = zs.try_iter()?.collect::<PyResult<_>>()?;
    let y_rows = read_rows(ys, None, q, m)?;
    let s_count = sk_hats.len();
    if s_count == 0 || y_rows.len() != s_count || z_items.len() != chunk_items.len() || ks.len() != chunk_items.len() {
        return Err(PyValueError::new_err("inconsistent decryption inputs"));
    }
    let eta = y_rows[0].len();
    if eta == 0 || y_rows.iter().any(|y| y.len() != eta) {
        return Err(fallback());
    }
    let dw = index_mod(delta, q, m)?;
    let bound_v: i128 = match py_index(bound)?.extract::<i64>() {
        Ok(b) => b as i128,
        Err(_) => if py_index(bound)?.gt(0)? { 1i128 << 126 } else { -1 },
    };
    // gather with the GIL: z per chunk, per slot eta rows of ct0 and ct1 (or bytes)
    let mut zg = Vec::with_capacity(z_items.len());
    let mut slots: Vec<Vec<Slot<'_>>> = Vec::with_capacity(chunk_items.len());
    for (j, (c, z)) in chunk_items.iter().zip(z_items.iter()).enumerate() {
        if ks[j] < 1 || ks[j] as usize > n {
            return Err(fallback());
        }
        zg.push(gather(z, n, max_len).ok_or_else(fallback)?);
        let per: Vec<Bound<'_, PyAny>> = if c.cast::<PyList>().is_ok() || c.cast::<PyTuple>().is_ok() {
            c.try_iter()?.collect::<PyResult<_>>()?
        } else {
            return Err(fallback());
        };
        if per.len() != s_count {
            return Err(fallback());
        }
        let mut row = Vec::with_capacity(s_count);
        for pair in per {
            if let Ok(b) = pair.cast::<PyBytes>() {
                row.push(Slot::Bin(b.clone()));
                continue;
            }
            let t = pair.cast::<PyTuple>().map_err(|_| fallback())?;
            if t.len() != 2 {
                return Err(fallback());
            }
            let (c0, c1) = (t.get_item(0)?, t.get_item(1)?);
            let rows: Vec<Bound<'_, PyAny>> = if c0.cast::<PyList>().is_ok() || c0.cast::<PyTuple>().is_ok() {
                c0.try_iter()?.collect::<PyResult<_>>()?
            } else {
                return Err(fallback());
            };
            if rows.len() != eta {
                return Err(fallback());
            }
            let mut g = Vec::with_capacity(eta + 1);
            for rr in rows.iter() {
                g.push(gather(rr, n, max_len).ok_or_else(fallback)?);
            }
            g.push(gather(&c1, n, max_len).ok_or_else(fallback)?);
            row.push(Slot::Dec(g));
        }
        slots.push(row);
    }
    // parse (without the GIL), then the values Python must read
    let qw = m.q;
    let parse = |g: &GText| -> Vec<Option<W>> {
        g.pos.iter().map(|&(p, l)| if p == SLOW { None } else { W::parse_dec(&g.text[p as usize..(p + l) as usize], qw) }).collect()
    };
    let ztexts: Vec<&GText> = zg.iter().map(|g| &g.t).collect();
    let mut zvals: Vec<Vec<Option<W>>> = py.detach(|| ztexts.iter().map(|t| parse(t)).collect());
    for (j, g) in zg.iter().enumerate() {
        for (i, v) in zvals[j].iter_mut().enumerate() {
            if v.is_none() {
                *v = Some(str_int_mod(&g.object(py, i), q)?); // int()'s ValueError, as gmpy2.mpz(z) would raise
            }
        }
    }
    // every gathered text of the slots, parsed in parallel without the GIL
    let texts: Vec<&GText> = slots.iter().flat_map(|row| row.iter()).flat_map(|slot| match slot {
        Slot::Dec(gs) => gs.iter().map(|g| &g.t).collect::<Vec<_>>(),
        Slot::Bin(_) => Vec::new(),
    }).collect();
    let parsed_cells: Vec<Mutex<Vec<Option<W>>>> = (0..texts.len()).map(|_| Mutex::new(Vec::new())).collect();
    py.detach(|| parallel_for(texts.len(), threads, |i| *parsed_cells[i].lock().unwrap() = parse(texts[i])));
    let mut parsed = parsed_cells.into_iter().map(|m| m.into_inner().unwrap());
    let mut slot_vals: Vec<Vec<SlotVals<'_, W>>> = Vec::with_capacity(slots.len());
    for (j, row) in slots.iter().enumerate() {
        let mut out_row = Vec::with_capacity(s_count);
        for (s, slot) in row.iter().enumerate() {
            let malformed = || PyValueError::new_err(("malformed ciphertext", s, j));
            let vals = match slot {
                Slot::Bin(b) => SlotVals::Bin(b.as_bytes()),
                Slot::Dec(gs) => {
                    let mut v = Vec::with_capacity((eta + 1) * n);
                    for g in gs {
                        for (i, p) in parsed.next().unwrap().into_iter().enumerate() {
                            v.push(match p {
                                Some(x) => x,
                                None => str_int_mod(&g.object(py, i), q).map_err(|_| malformed())?,
                            });
                        }
                    }
                    SlotVals::Words(v)
                }
            };
            out_row.push(vals);
        }
        slot_vals.push(out_row);
    }
    let total: usize = ks.iter().map(|&k| k as usize).sum();
    let mut values = vec![0i64; total];
    let mut none = vec![0u8; total];
    let starts: Vec<usize> = ks.iter().scan(0usize, |acc, &k| { let s = *acc; *acc += k as usize; Some(s) }).collect();
    let ym: Vec<W> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
    let hats: Vec<Vec<W>> = sk_hats.iter().map(|h| bytes_to_words(h, n)).collect::<PyResult<_>>()?;
    let zw: Vec<Vec<W>> = zvals.into_iter().map(|z| z.into_iter().map(|v| v.unwrap()).collect()).collect();
    let logn = r.logn;
    let outs: Vec<Mutex<(Vec<i64>, Vec<u8>, bool)>> = ks.iter().map(|&k| Mutex::new((vec![0; k as usize], vec![0; k as usize], true))).collect();
    // first malformed binary slot of each chunk
    let bad: Vec<Mutex<Option<usize>>> = (0..slot_vals.len()).map(|_| Mutex::new(None)).collect();
    py.detach(|| parallel_for(slot_vals.len(), threads, |j| {
        let mut buf = vec![W::ZERO; (eta + 1) * n];
        let mut words = Vec::with_capacity(s_count);
        for (s, sv) in slot_vals[j].iter().enumerate() {
            words.push(match sv {
                SlotVals::Words(v) => v.clone(),
                SlotVals::Bin(b) => {
                    if bin_read(b, qw, logn, eta, &mut buf).is_none() {
                        *bad[j].lock().unwrap() = Some(s);
                        return;
                    }
                    buf.clone()
                }
            });
        }
        let slot_refs: Vec<&[W]> = words.iter().map(|v| v.as_slice()).collect();
        let hat_refs: Vec<&[W]> = hats.iter().map(|v| v.as_slice()).collect();
        let mut o = outs[j].lock().unwrap();
        let (ref mut vals, ref mut nn, ref mut ok) = *o;
        *ok = rlwe::decrypt_chunk(r, eta, &zw[j], &slot_refs, &ym, &hat_refs, dw, bound_v, vals, nn);
    }));
    if let Some((j, s)) = bad.into_iter().enumerate().find_map(|(j, b)| b.into_inner().unwrap().map(|s| (j, s))) {
        return Err(PyValueError::new_err(("malformed ciphertext", s, j)));
    }
    let mut big = false;
    for (j, o) in outs.into_iter().enumerate() {
        let (v, nn, ok) = o.into_inner().unwrap();
        big |= !ok;
        values[starts[j]..starts[j] + v.len()].copy_from_slice(&v);
        none[starts[j]..starts[j] + nn.len()].copy_from_slice(&nn);
    }
    if big {
        return Err(fallback()); // |values| beyond int64: let Python decode them
    }
    let data = PyBytes::new(py, &values.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>());
    let mask: Py<PyAny> = if none.iter().any(|&b| b == 1) { PyBytes::new(py, &none).into_any().unbind() } else { py.None() };
    Ok(PyTuple::new(py, [data.into_any().unbind(), mask])?.into_any().unbind())
}

/// MCFE key terms z'' of several labels as decimal strings.
#[pyfunction]
fn pad_terms(py: Python<'_>, ctx: &Bound<'_, RingCtx>, base: &Bound<'_, PyAny>, materials: &Bound<'_, PyAny>,
             ys: &Bound<'_, PyAny>, eta: usize, threads: usize) -> PyResult<Py<PyList>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let n = ring.n();
    with_ring!(&*ring, r => {
        let m = &r.m;
        let basev = read_values(base, Some(n), &q, m, false)?;
        let y_rows = read_rows(ys, Some(eta), &q, m)?;
        let mats: Vec<Vec<Vec<u8>>> = materials.try_iter()?.map(|row| {
            let row = row?;
            let v: Vec<Vec<u8>> = row.try_iter()?.map(|m| m?.cast::<PyBytes>().map(|b| b.as_bytes().to_vec())
                .map_err(|_| PyTypeError::new_err("materials must be bytes"))).collect::<PyResult<_>>()?;
            if v.len() != y_rows.len() {
                return Err(PyValueError::new_err("one material per slot and label"));
            }
            Ok(v)
        }).collect::<PyResult<_>>()?;
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
        let refs: Vec<Vec<&[u8]>> = mats.iter().map(|row| row.iter().map(|v| v.as_slice()).collect()).collect();
        let outs: Vec<Mutex<Option<DecBuf>>> = (0..refs.len()).map(|_| Mutex::new(None)).collect();
        py.detach(|| rlwe::pad_terms(r, &basev, &refs, &ym, eta, threads, |l, z| *outs[l].lock().unwrap() = Some(DecBuf::new(z))));
        let lists = outs.into_iter().map(|o| o.into_inner().unwrap().unwrap().list(py)).collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, lists)?.unbind())
    })
}

// ------------------------------------------------------ offline / online

enum AnyPre {
    W64(Precomputed<u64>),
    W128(Precomputed<u128>),
    W256(Precomputed<U256>),
}

/// The plaintext-independent part of an MCFE ndarray encryption (one-time).
#[pyclass(module = "pyfe4ai._rust_native")]
struct PrecomputedObj {
    key: Py<EncKeyObj>,
    pre: Mutex<AnyPre>,
}

#[pyfunction]
#[pyo3(signature = (key, pad_materials, sigma2, sigma3, rnd, threads, seed=None, facct=None))]
#[allow(clippy::too_many_arguments)]
fn precompute_tensor(py: Python<'_>, key: &Bound<'_, EncKeyObj>, pad_materials: &Bound<'_, PyAny>, sigma2: f64,
                     sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, seed: Option<Vec<u8>>, facct: FacctArg)
    -> PyResult<PrecomputedObj> {
    let obj = key.get();
    let mats = materials_of(pad_materials)?.unwrap_or_default();
    let refs: Vec<&[u8]> = mats.iter().map(|v| v.as_slice()).collect();
    let rb: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    let stream = |c: usize, off: u64, buf: &mut [u8]| drbg.fill_at(c as u64, off, buf);
    macro_rules! run {
        ($r:expr, $k:expr, $variant:ident) => {
            with_noise!(py, facct, noise, err, {
                AnyPre::$variant(py.detach(|| rlwe::precompute($r, $k, &refs, sigma2, sigma3, &noise, randomness(&rb, &stream),
                    threads)).map_err(core_err)?)
            })
        };
    }
    let pre = match (&*obj.ring, &obj.key) {
        (AnyRing::W64(r), AnyKey::W64(k)) => run!(r, k, W64),
        (AnyRing::W128(r), AnyKey::W128(k)) => run!(r, k, W128),
        (AnyRing::W256(r), AnyKey::W256(k)) => run!(r, k, W256),
        _ => unreachable!(),
    };
    Ok(PrecomputedObj { key: key.clone().unbind(), pre: Mutex::new(pre) })
}

#[pyfunction]
#[pyo3(signature = (key, pre, first, x, threads, binary=false))]
fn encrypt_precomputed(py: Python<'_>, key: &Bound<'_, EncKeyObj>, pre: &Bound<'_, PrecomputedObj>, first: usize,
                       x: &Bound<'_, PyAny>, threads: usize, binary: bool) -> PyResult<Py<PyList>> {
    let p = pre.borrow();
    if !p.key.bind(py).is(key) {
        return Err(PyValueError::new_err("precomputed with another key"));
    }
    let xs = int64_buffer(x)?;
    let obj = key.get();
    let mut guard = p.pre.lock().unwrap();
    macro_rules! run {
        ($r:expr, $k:expr, $pc:expr) => {{
            let n = $r.n;
            let chunks = (xs.len() + n - 1) / n;
            let outs: Vec<Mutex<Option<ChunkOut>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
            py.detach(|| rlwe::encrypt_precomputed($r, $k, $pc, first, &xs, threads,
                |j, c0, c1| *outs[j].lock().unwrap() = Some(chunk_out($r, binary, c0, c1)))).map_err(core_err)?;
            chunk_list(py, outs.into_iter().map(|m| m.into_inner().unwrap()).collect(), n, xs.len())
        }};
    }
    match (&*obj.ring, &obj.key, &mut *guard) {
        (AnyRing::W64(r), AnyKey::W64(k), AnyPre::W64(pc)) => run!(r, k, pc),
        (AnyRing::W128(r), AnyKey::W128(k), AnyPre::W128(pc)) => run!(r, k, pc),
        (AnyRing::W256(r), AnyKey::W256(k), AnyPre::W256(pc)) => run!(r, k, pc),
        _ => Err(PyValueError::new_err("precomputed with another key")),
    }
}

#[pyfunction]
fn precomputed_info(py: Python<'_>, pre: &Bound<'_, PrecomputedObj>) -> PyResult<Py<PyAny>> {
    let p = pre.borrow();
    let g = p.pre.lock().unwrap();
    let (chunks, used) = match &*g {
        AnyPre::W64(x) => (x.chunks(), x.used.iter().filter(|&&u| u).count()),
        AnyPre::W128(x) => (x.chunks(), x.used.iter().filter(|&&u| u).count()),
        AnyPre::W256(x) => (x.chunks(), x.used.iter().filter(|&&u| u).count()),
    };
    let d = pyo3::types::PyDict::new(py);
    d.set_item("chunks", chunks)?;
    d.set_item("used", used)?;
    Ok(d.into_any().unbind())
}

// ------------------------------------------------------- threshold MCFE

/// Values of a list or tuple of n items mod q, each a plain decimal string
/// (int(str(v)) semantics) or an exact int; None for anything that needs
/// Python's int() (other types, signs, blanks, underscores, other digits).
fn read_plain<W: Word>(seq: &Bound<'_, PyAny>, n: usize, m: &Modulus<W>) -> Option<Vec<W>> {
    let g = gather(seq, n, W::DEC_DIGITS + 1)?;
    let mut out = Vec::with_capacity(n);
    let mut slow = g.slow.iter();
    for &(p, l) in g.t.pos.iter() {
        out.push(if p == SLOW {
            int_mod_fast(&slow.next()?.1, m)?
        } else {
            W::parse_dec(&g.t.text[p as usize..(p + l) as usize], m.q)?
        });
    }
    Some(out)
}

fn join_dec<W: Word>(vals: &[W], out: &mut Vec<u8>) {
    out.push(b'[');
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b", ");
        }
        v.format_dec(out);
    }
    out.push(b']');
}

/// The clients' ciphertexts of one chunk as canonical_ring_ciphertext reads
/// them, and the JSON texts ciphertexts_digest hashes for {nid: (k, ct0, ct1)}
/// and {nid: ct1}: `keys` are the JSON-encoded client ids in sorted order, `ks`
/// their batch lengths, `cts` their (ct0 rows, ct1) in that order. Returns
/// (packed words per client: the eta rows of ct0, then ct1; text; text), or
/// None when a value or a layout needs Python (see read_plain).
#[pyfunction]
fn share_inputs(py: Python<'_>, ctx: &Bound<'_, RingCtx>, keys: Vec<String>, ks: Vec<i64>, cts: &Bound<'_, PyAny>,
                eta: usize) -> PyResult<Option<(Vec<Py<PyBytes>>, Py<PyBytes>, Py<PyBytes>)>> {
    let ring = get_ring(ctx);
    let items: Vec<Bound<'_, PyAny>> = cts.try_iter()?.collect::<PyResult<_>>()?;
    if items.len() != keys.len() || ks.len() != keys.len() {
        return Err(PyValueError::new_err("one key and batch length per ciphertext"));
    }
    with_ring!(&*ring, r => {
        let n = r.n;
        let mut words = Vec::with_capacity(items.len());
        for it in items.iter() {
            let t = match it.cast::<PyTuple>() {
                Ok(t) if t.len() == 2 => t,
                _ => return Err(PyValueError::new_err("each ciphertext as (ct0 rows, ct1)")),
            };
            let (c0, c1) = (t.get_item(0)?, t.get_item(1)?);
            let rows: Vec<Bound<'_, PyAny>> = match (c0.cast::<PyList>(), c0.cast::<PyTuple>()) {
                (Ok(l), _) => l.iter().collect(),
                (_, Ok(t)) => t.iter().collect(),
                _ => return Ok(None),
            };
            if rows.len() != eta {
                return Ok(None);
            }
            let mut v = Vec::with_capacity((eta + 1) * n);
            for row in rows.iter().chain(std::iter::once(&c1)) {
                match read_plain(row, n, &r.m) {
                    Some(x) => v.extend_from_slice(&x),
                    None => return Ok(None),
                }
            }
            words.push(v);
        }
        let (ct_text, key_text) = py.detach(|| {
            let per = n * (eta + 1) * (r.m.bits as usize * 3 / 10 + 3);
            let (mut a, mut b) = (Vec::with_capacity(per * words.len() + 64), Vec::with_capacity(per * words.len() / (eta + 1) + 64));
            a.push(b'{');
            b.push(b'{');
            for (c, (key, v)) in keys.iter().zip(words.iter()).enumerate() {
                if c > 0 {
                    a.extend_from_slice(b", ");
                    b.extend_from_slice(b", ");
                }
                a.extend_from_slice(key.as_bytes());
                a.extend_from_slice(b": [");
                a.extend_from_slice(ks[c].to_string().as_bytes());
                a.extend_from_slice(b", [");
                for row in 0..eta {
                    if row > 0 {
                        a.extend_from_slice(b", ");
                    }
                    join_dec(&v[row * n..(row + 1) * n], &mut a);
                }
                a.extend_from_slice(b"], ");
                join_dec(&v[eta * n..], &mut a);
                a.push(b']');
                b.extend_from_slice(key.as_bytes());
                b.extend_from_slice(b": ");
                join_dec(&v[eta * n..], &mut b);
            }
            a.push(b'}');
            b.push(b'}');
            (a, b)
        });
        let packed = words.iter().map(|v| words_to_bytes(py, v)).collect();
        Ok(Some((packed, PyBytes::new(py, &ct_text).unbind(), PyBytes::new(py, &key_text).unbind())))
    })
}

/// ThresholdMCFERingLWE.share_decrypt of one chunk with its noise drawn: (w, c_y)
/// as decimal strings, from share_inputs' packed ciphertexts, the weights (one
/// row of eta per client), the NTT(sk share)s of ntt_mont, the z share, Delta,
/// the Lagrange coefficient and the noise (n values).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn share_partial(py: Python<'_>, ctx: &Bound<'_, RingCtx>, packed: Vec<Vec<u8>>, eta: usize, ys: &Bound<'_, PyAny>,
                 sk_hats: Vec<Vec<u8>>, z: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>, lagrange: &Bound<'_, PyAny>,
                 noise: &Bound<'_, PyAny>) -> PyResult<(Py<PyList>, Py<PyList>)> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    if packed.is_empty() || packed.len() != sk_hats.len() || eta == 0 {
        return Err(PyValueError::new_err("one ciphertext and key share per client"));
    }
    with_ring!(&*ring, r => {
        let (n, m) = (r.n, &r.m);
        let slots: Vec<Vec<_>> = packed.iter().map(|b| bytes_to_words(b, (eta + 1) * n)).collect::<PyResult<_>>()?;
        let hats: Vec<Vec<_>> = sk_hats.iter().map(|b| bytes_to_words(b, n)).collect::<PyResult<_>>()?;
        let y_rows = read_rows(ys, Some(eta), &q, m)?;
        if y_rows.len() != slots.len() {
            return Err(PyValueError::new_err("one weight row per client"));
        }
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
        let zv = read_values(z, Some(n), &q, m, false)?;
        let ev = read_values(noise, Some(n), &q, m, false)?;
        let (d, l) = (index_mod(delta, &q, m)?, index_mod(lagrange, &q, m)?);
        let (mut w, mut cy) = (vec![Word::ZERO; n], vec![Word::ZERO; n]);
        py.detach(|| {
            let sr: Vec<&[_]> = slots.iter().map(|v| v.as_slice()).collect();
            let hr: Vec<&[_]> = hats.iter().map(|v| v.as_slice()).collect();
            rlwe::share_partial(r, eta, &sr, &ym, &hr, &zv, d, l, &ev, &mut w, &mut cy);
        });
        Ok((DecBuf::new(&w).list(py)?.unbind(), DecBuf::new(&cy).list(py)?.unbind()))
    })
}

/// ThresholdMCFERingLWE._combine_group: decode_scaled of c_y - sum_k w_k mod q
/// for the first k values (ints, None where |mu| > bound); None when a value
/// needs Python (see read_plain) or a result does not fit an int64.
#[pyfunction]
fn share_combine(py: Python<'_>, ctx: &Bound<'_, RingCtx>, c_y: &Bound<'_, PyAny>, ws: &Bound<'_, PyAny>,
                 delta: &Bound<'_, PyAny>, bound: &Bound<'_, PyAny>, k: usize) -> PyResult<Option<Py<PyList>>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let bound_v: i128 = match py_index(bound)?.extract::<i64>() {
        Ok(b) => b as i128,
        Err(_) => if py_index(bound)?.gt(0)? { 1i128 << 126 } else { -1 },
    };
    let w_items: Vec<Bound<'_, PyAny>> = ws.try_iter()?.collect::<PyResult<_>>()?;
    with_ring!(&*ring, r => {
        let (n, m) = (r.n, &r.m);
        if k < 1 || k > n {
            return Ok(None);
        }
        let cy = match read_plain(c_y, n, m) {
            Some(v) => v,
            None => return Ok(None),
        };
        let mut wv = Vec::with_capacity(w_items.len());
        for w in w_items.iter() {
            match read_plain(w, n, m) {
                Some(v) => wv.push(v),
                None => return Ok(None),
            }
        }
        let d = index_mod(delta, &q, m)?;
        let (mut out, mut none) = (vec![0i64; k], vec![0u8; k]);
        let wr: Vec<&[_]> = wv.iter().map(|v| v.as_slice()).collect();
        if !py.detach(|| rlwe::share_combine(m, &cy, &wr, d, bound_v, &mut out, &mut none)) {
            return Ok(None);
        }
        let items = out.iter().zip(none.iter())
            .map(|(&v, &nn)| if nn == 1 { Ok(py.None().into_bound(py)) } else { Ok(v.into_pyobject(py)?.into_any()) })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some(PyList::new(py, items)?.unbind()))
    })
}

/// threshold_utils.share_secret for many secrets: one list of shares (decimal
/// strings) per point. The coefficients are drawn as SystemRandom.randrange(0, q)
/// draws them, from the given bytes (tests; ValueError if they run out) or the
/// call's DRBG (stream 0, read sequentially).
#[pyfunction]
#[pyo3(signature = (ctx, values, t, points, rnd, seed=None))]
fn shamir_share(py: Python<'_>, ctx: &Bound<'_, RingCtx>, values: &Bound<'_, PyAny>, t: usize, points: Vec<u64>,
                rnd: &Bound<'_, PyAny>, seed: Option<Vec<u8>>) -> PyResult<Py<PyList>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    if t < 1 {
        return Err(PyValueError::new_err("the threshold must be at least 1"));
    }
    let given: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    with_ring!(&*ring, r => {
        let m = &r.m;
        let secrets = read_values(values, None, &q, m, false)?;
        let shares = py.detach(|| match &given {
            Some(b) => {
                let mut pos = 0;
                pyfe4ai_core::shamir::share(m, &secrets, t, &points, |out: &mut [u8]| {
                    if pos + out.len() > b.len() {
                        return false;
                    }
                    out.copy_from_slice(&b[pos..pos + out.len()]);
                    pos += out.len();
                    true
                })
            }
            None => {
                let mut rng = drbg.rng(0);
                let res = pyfe4ai_core::shamir::share(m, &secrets, t, &points, |out: &mut [u8]| {
                    rng.fill_bytes(out);
                    true
                });
                wipe_rng(&mut rng);
                res
            }
        });
        let shares = shares.ok_or_else(|| PyValueError::new_err("not enough random bytes"))?;
        let lists = shares.iter().map(|s| DecBuf::new(s).list(py)).collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, lists)?.unbind())
    })
}

// --------------------------------------------------------------- dense LWE

#[pyfunction]
fn dense_dots(py: Python<'_>, materials: &Bound<'_, PyAny>, nb: usize, count: usize, key: &[u8], kl: usize,
              threads: usize) -> PyResult<Py<PyBytes>> {
    let mats = materials_of(materials)?.unwrap_or_default();
    if nb == 0 || kl == 0 || key.len() != count * kl * 8 {
        return Err(PyValueError::new_err("inconsistent dense_dots arguments"));
    }
    let k: Vec<u64> = key.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let refs: Vec<&[u8]> = mats.iter().map(|v| v.as_slice()).collect();
    let sums = py.detach(|| pyfe4ai_core::dense::dense_dots(&refs, nb, count, &k, kl, threads));
    Ok(PyBytes::new(py, &sums.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unbind())
}

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

#[pymodule]
fn _rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RingCtx>()?;
    m.add_class::<EncKeyObj>()?;
    m.add_class::<PrecomputedObj>()?;
    m.add_class::<FacctObj>()?;
    for f in [
        wrap_pyfunction!(ctx_new, m)?, wrap_pyfunction!(ctx_info, m)?, wrap_pyfunction!(pack, m)?,
        wrap_pyfunction!(unpack, m)?, wrap_pyfunction!(ntt, m)?, wrap_pyfunction!(ntt_mont, m)?,
        wrap_pyfunction!(poly_mul, m)?, wrap_pyfunction!(lincomb, m)?, wrap_pyfunction!(gauss, m)?,
        wrap_pyfunction!(shake256_py, m)?, wrap_pyfunction!(hash_to_zq_py, m)?, wrap_pyfunction!(random_bytes, m)?,
        wrap_pyfunction!(enc_key, m)?, wrap_pyfunction!(encrypt, m)?, wrap_pyfunction!(encrypt_tensor, m)?, wrap_pyfunction!(facct_sample, m)?,
        wrap_pyfunction!(decrypt_tensor, m)?, wrap_pyfunction!(pad_terms, m)?,
        wrap_pyfunction!(precompute_tensor, m)?, wrap_pyfunction!(encrypt_precomputed, m)?,
        wrap_pyfunction!(precomputed_info, m)?, wrap_pyfunction!(dense_dots, m)?, wrap_pyfunction!(build_info, m)?,
        wrap_pyfunction!(drbg_bytes, m)?, wrap_pyfunction!(share_inputs, m)?, wrap_pyfunction!(share_partial, m)?,
        wrap_pyfunction!(share_combine, m)?, wrap_pyfunction!(shamir_share, m)?,
    ] {
        m.add_function(f)?;
    }
    ec::register(m)?;
    pairing::register(m)?;
    Ok(())
}
