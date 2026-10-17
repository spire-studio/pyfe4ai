//! pyfe4ai._rust_native: the Rust backend of pyfe4ai.utils.native_backend.
//!
//! The lattice functions have the names, arguments, results and errors of the
//! C++ module pyfe4ai._rlwe_native, so the same Python glue and the same
//! bit-identity tests run on either; `ec` adds ristretto255. Values cross the
//! boundary as Python ints, decimal strings (the ciphertext format) or packed
//! little-endian words in bytes; the GIL is released around the kernels.

use pyfe4ai_core::codec::{bin_read, bin_size, bin_wb, bin_write, format_dec, parse_dec};
use pyfe4ai_core::gauss::{gauss_bytes, gauss_fill};
use pyfe4ai_core::keccak::{hash_to_zq, shake256};
use pyfe4ai_core::rlwe::{self, EncKey, Precomputed, Randomness};
use pyfe4ai_core::{osrand, parallel_for, with_ring, AnyRing, Ring, Word};
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::sync::{Arc, Mutex};

mod ec;

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

    /// Bytes of stream `stream` from its start.
    fn fill(&self, stream: u64, out: &mut [u8]) {
        let mut rng = ChaCha20Rng::from_seed(*self.seed);
        rng.set_stream(stream);
        rng.fill_bytes(out);
        // ChaCha20Rng holds only integers: zero its bytes before it is dropped
        unsafe {
            let p = &mut rng as *mut ChaCha20Rng as *mut u8;
            for i in 0..std::mem::size_of::<ChaCha20Rng>() {
                std::ptr::write_volatile(p.add(i), 0);
            }
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
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

fn u128_arg(ob: &Bound<'_, PyAny>) -> PyResult<u128> {
    let v = py_index(ob)?;
    v.extract::<u128>().map_err(|_| PyValueError::new_err("needs a non-negative 128-bit integer"))
}

fn py_index<'py>(ob: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    ob.py().import("operator")?.getattr("index")?.call1((ob,))
}

/// int(str(ob)) % q (the semantics of canonical_ring_ciphertext).
fn str_int_mod(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<u128> {
    let int = ob.py().import("builtins")?.getattr("int")?;
    int.call1((ob.str()?,))?.rem(q)?.extract::<u128>()
}

/// operator.index(ob) % q.
fn index_mod(ob: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<u128> {
    py_index(ob)?.rem(q)?.extract::<u128>()
}

/// Values of a sequence mod q: decimal strings by the fast path, other strings
/// as int(s), anything else by __index__ (str_semantics: int(str(v)) for all).
fn read_values(seq: &Bound<'_, PyAny>, n: Option<usize>, q: &Bound<'_, PyAny>, qv: u128, str_semantics: bool)
    -> PyResult<Vec<u128>> {
    let items: Vec<Bound<'_, PyAny>> = seq.try_iter()?.collect::<PyResult<_>>()?;
    if let Some(n) = n {
        if items.len() != n {
            return Err(PyValueError::new_err(format!("expected {} values", n)));
        }
    }
    items
        .iter()
        .map(|ob| {
            if let Ok(s) = ob.cast::<PyString>() {
                if let Some(v) = s.to_str().ok().and_then(|t| parse_dec(t.as_bytes(), qv)) {
                    return Ok(v);
                }
                return str_int_mod(ob, q);
            }
            if str_semantics { str_int_mod(ob, q) } else { index_mod(ob, q) }
        })
        .collect()
}

fn get_ring(ctx: &Bound<'_, RingCtx>) -> Arc<AnyRing> {
    ctx.get().ring.clone()
}

#[pyfunction]
fn ctx_new(py: Python<'_>, q: &Bound<'_, PyAny>, n: usize, root: &Bound<'_, PyAny>) -> PyResult<RingCtx> {
    let (qv, rv) = (u128_arg(q)?, u128_arg(root)?);
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
    let mut out = Vec::with_capacity(w.len() * (W::BITS as usize / 8));
    for v in w {
        out.extend_from_slice(&v.to_u128().to_le_bytes()[..W::BITS as usize / 8]);
    }
    PyBytes::new(py, &out).unbind()
}

fn bytes_to_words<W: Word>(b: &[u8], count: usize) -> PyResult<Vec<W>> {
    let wb = W::BITS as usize / 8;
    if b.len() != count * wb {
        return Err(PyValueError::new_err(format!("expected {} bytes of words", count * wb)));
    }
    Ok(b.chunks_exact(wb)
        .map(|c| {
            let mut buf = [0u8; 16];
            buf[..wb].copy_from_slice(c);
            W::from_u128(u128::from_le_bytes(buf))
        })
        .collect())
}

/// A str of ASCII bytes, without UTF-8 decoding (as the C++ module builds them).
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
    let mut buf = Vec::with_capacity(40);
    let items = vals
        .iter()
        .map(|v| {
            buf.clear();
            format_dec(v.to_u128(), &mut buf);
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
        let mut text = Vec::with_capacity(vals.len() * 24);
        let mut off = Vec::with_capacity(vals.len() + 1);
        for v in vals {
            off.push(text.len() as u32);
            format_dec(v.to_u128(), &mut text);
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
    let vals = read_values(values, None, q, ring.q(), false)?;
    with_ring!(&*ring, r => {
        let w: Vec<_> = vals.iter().map(|&v| r.m.reduce_u128(v)).collect();
        Ok(words_to_bytes(py, &w))
    })
}

#[pyfunction]
#[pyo3(signature = (ctx, data, as_digits=false))]
fn unpack(py: Python<'_>, ctx: &Bound<'_, RingCtx>, data: &[u8], as_digits: bool) -> PyResult<Py<PyList>> {
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let _ = r;
        let size = r_word_bytes(&ring);
        if data.len() % size != 0 {
            return Err(PyValueError::new_err("length is not a multiple of the word size"));
        }
        let w: Vec<u128> = data.chunks_exact(size).map(|c| {
            let mut buf = [0u8; 16];
            buf[..size].copy_from_slice(c);
            u128::from_le_bytes(buf)
        }).collect();
        if as_digits {
            return Ok(dec_list(py, &w)?.unbind());
        }
        Ok(PyList::new(py, w)?.unbind())
    })
}

fn r_word_bytes(r: &AnyRing) -> usize {
    r.word_bits() as usize / 8
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
    let ys = read_values(coeffs, None, q, ring.q(), false)?;
    let rows: Vec<Vec<u8>> = rows.try_iter()?.map(|r| r?.extract::<Vec<u8>>()).collect::<PyResult<_>>()?;
    if rows.len() != ys.len() {
        return Err(PyValueError::new_err("one coefficient per row"));
    }
    with_ring!(&*ring, r => {
        let mut acc = vec![Word::ZERO; r.n];
        for (row, &y) in rows.iter().zip(ys.iter()) {
            let v = bytes_to_words(row, r.n)?;
            let ym = r.m.to_mont(r.m.reduce_u128(y));
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
}

/// A slot's encryption key: NTT(a), NTT(pk_row) in Montgomery form, u, Delta.
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
struct EncKeyObj {
    ring: Arc<AnyRing>,
    key: AnyKey,
}

macro_rules! with_key {
    ($obj:expr, $r:ident, $k:ident => $body:expr) => {
        match (&*$obj.ring, &$obj.key) {
            (AnyRing::W64($r), AnyKey::W64($k)) => $body,
            (AnyRing::W128($r), AnyKey::W128($k)) => $body,
            _ => unreachable!(),
        }
    };
}

#[pyfunction]
fn enc_key(py: Python<'_>, ctx: &Bound<'_, RingCtx>, a: &Bound<'_, PyAny>, pk: &Bound<'_, PyAny>,
           u: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>) -> PyResult<EncKeyObj> {
    let q = ctx.get().q.bind(py);
    let ring = get_ring(ctx);
    let (n, qv) = (ring.n(), ring.q());
    let av = read_values(a, Some(n), q, qv, false)?;
    let pk_rows: Vec<Vec<u128>> = pk.try_iter()?.map(|row| read_values(&row?, Some(n), q, qv, false))
        .collect::<PyResult<_>>()?;
    if pk_rows.is_empty() {
        return Err(PyValueError::new_err("pk needs at least one row"));
    }
    let u_rows: Option<Vec<Vec<u128>>> = if u.is_none() {
        None
    } else {
        let rows: Vec<Vec<u128>> = u.try_iter()?.map(|row| read_values(&row?, Some(n), q, qv, false))
            .collect::<PyResult<_>>()?;
        if rows.len() != pk_rows.len() {
            return Err(PyValueError::new_err("one pad row per pk row"));
        }
        Some(rows)
    };
    let d = index_mod(delta, q)?;
    macro_rules! build {
        ($r:expr, $variant:ident) => {{
            let conv = |v: &Vec<u128>| v.iter().map(|&x| $r.m.reduce_u128(x)).collect::<Vec<_>>();
            let pkw: Vec<_> = pk_rows.iter().map(conv).collect();
            let uw: Option<Vec<_>> = u_rows.as_ref().map(|rows| rows.iter().map(conv).collect());
            AnyKey::$variant(EncKey::new($r, &conv(&av), &pkw, uw.as_deref(), $r.m.reduce_u128(d)))
        }};
    }
    let key = match &*ring {
        AnyRing::W64(r) => build!(r, W64),
        AnyRing::W128(r) => build!(r, W128),
    };
    Ok(EncKeyObj { ring, key })
}

/// Given bytes (tests), else the call's DRBG streams.
fn randomness<'a>(rnd: &'a Option<Vec<u8>>, stream: &'a (dyn Fn(usize, &mut [u8]) + Sync)) -> Randomness<'a> {
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

/// MIFERingLWE._encrypt_rows: (ct0, ct1) as decimal strings, or format-1 bytes.
#[pyfunction]
#[pyo3(signature = (key, w_rows, sigma2, sigma3, rnd, binary=false, seed=None))]
#[allow(clippy::too_many_arguments)]
fn encrypt(py: Python<'_>, key: &Bound<'_, EncKeyObj>, w_rows: &Bound<'_, PyAny>, sigma2: f64, sigma3: f64,
           rnd: &Bound<'_, PyAny>, binary: bool, seed: Option<Vec<u8>>) -> PyResult<Py<PyAny>> {
    let obj = key.get();
    let ring = obj.ring.clone();
    let (n, qv) = (ring.n(), ring.q());
    let qob = PyModule::import(py, "builtins")?.getattr("int")?.call1((qv,))?;
    let rows: Vec<Vec<u128>> = w_rows.try_iter()?.map(|row| read_values(&row?, Some(n), &qob, qv, false))
        .collect::<PyResult<_>>()?;
    with_key!(obj, r, k => {
        let eta = k.eta;
        if rows.len() != eta {
            return Err(PyValueError::new_err("one plaintext row per pk row"));
        }
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
        let w: Vec<_> = rows.iter().flatten().map(|&v| r.m.reduce_u128(v)).collect();
        let (mut c0, mut c1) = (vec![Word::ZERO; eta * n], vec![Word::ZERO; n]);
        py.detach(|| rlwe::encrypt_chunk(r, k, &w, sigma2, sigma3, &rb, &mut c0, &mut c1));
        rlwe::wipe_bytes(&mut rb);
        if binary {
            let mut out = Vec::new();
            let c0u: Vec<u128> = c0.iter().map(|v| v.to_u128()).collect();
            let c1u: Vec<u128> = c1.iter().map(|v| v.to_u128()).collect();
            bin_write(ring.qbits(), r.logn, eta, &c0u, &c1u, &mut out);
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
        let (a, b): (Vec<u128>, Vec<u128>) = (c0.iter().map(|v| v.to_u128()).collect(), c1.iter().map(|v| v.to_u128()).collect());
        let mut out = Vec::with_capacity(bin_size(1, r.n, bin_wb(r.m.bits)));
        bin_write(r.m.bits, r.logn, 1, &a, &b, &mut out);
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
#[pyo3(signature = (key, x, pad_materials, sigma2, sigma3, rnd, threads, binary=false, seed=None))]
#[allow(clippy::too_many_arguments)]
fn encrypt_tensor(py: Python<'_>, key: &Bound<'_, EncKeyObj>, x: &Bound<'_, PyAny>, pad_materials: &Bound<'_, PyAny>,
                  sigma2: f64, sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, binary: bool,
                  seed: Option<Vec<u8>>) -> PyResult<Py<PyList>> {
    let obj = key.get();
    let xs = int64_buffer(x)?;
    let mats = materials_of(pad_materials)?;
    let rb: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    let stream = |c: usize, buf: &mut [u8]| drbg.fill(c as u64, buf);
    with_key!(obj, r, k => {
        let n = r.n;
        let chunks = if xs.is_empty() { 0 } else { (xs.len() + n - 1) / n };
        let outs: Vec<Mutex<Option<ChunkOut>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
        let refs: Option<Vec<&[u8]>> = mats.as_ref().map(|m| m.iter().map(|v| v.as_slice()).collect());
        py.detach(|| rlwe::encrypt_tensor(r, k, &xs, refs.as_deref(), sigma2, sigma3, randomness(&rb, &stream), threads,
            |j, c0, c1| *outs[j].lock().unwrap() = Some(chunk_out(r, binary, c0, c1))))
            .map_err(core_err)?;
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

fn gather<'py>(seq: &Bound<'py, PyAny>, n: usize) -> Option<Gathered<'py>> {
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
                && pyo3::ffi::PyUnicode_GET_LENGTH(p) <= 39
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

enum Slot<'py> {
    Dec(Vec<Gathered<'py>>), // eta rows of ct0, then ct1
    Bin(Vec<u8>),
}

/// MIFERingLWE._decrypt_with_pad_term for every chunk of a layer (see the C++
/// decrypt_tensor): returns (int64 bytes, None mask bytes | None).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decrypt_tensor(py: Python<'_>, ctx: &Bound<'_, RingCtx>, chunks: &Bound<'_, PyAny>, zs: &Bound<'_, PyAny>,
                  ks: Vec<i64>, sk_hats: Vec<Vec<u8>>, ys: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>,
                  bound: &Bound<'_, PyAny>, threads: usize) -> PyResult<Py<PyAny>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let (n, qv) = (ring.n(), ring.q());
    let fallback = || PyNotImplementedError::new_err("input layout not handled natively");
    let chunk_items: Vec<Bound<'_, PyAny>> = chunks.try_iter()?.collect::<PyResult<_>>()?;
    let z_items: Vec<Bound<'_, PyAny>> = zs.try_iter()?.collect::<PyResult<_>>()?;
    let y_rows: Vec<Vec<u128>> = ys.try_iter()?.map(|y| read_values(&y?, None, &q, qv, false)).collect::<PyResult<_>>()?;
    let s_count = sk_hats.len();
    if s_count == 0 || y_rows.len() != s_count || z_items.len() != chunk_items.len() || ks.len() != chunk_items.len() {
        return Err(PyValueError::new_err("inconsistent decryption inputs"));
    }
    let eta = y_rows[0].len();
    if eta == 0 || y_rows.iter().any(|y| y.len() != eta) {
        return Err(fallback());
    }
    let d = index_mod(delta, &q)?;
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
        zg.push(gather(z, n).ok_or_else(fallback)?);
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
                row.push(Slot::Bin(b.as_bytes().to_vec()));
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
                g.push(gather(rr, n).ok_or_else(fallback)?);
            }
            g.push(gather(&c1, n).ok_or_else(fallback)?);
            row.push(Slot::Dec(g));
        }
        slots.push(row);
    }
    // parse (without the GIL), then the values Python must read
    let parse = |g: &GText| -> Vec<Option<u128>> {
        g.pos.iter().map(|&(p, l)| if p == SLOW { None } else { parse_dec(&g.text[p as usize..(p + l) as usize], qv) }).collect()
    };
    let ztexts: Vec<&GText> = zg.iter().map(|g| &g.t).collect();
    let mut zvals: Vec<Vec<Option<u128>>> = py.detach(|| ztexts.iter().map(|t| parse(t)).collect());
    for (j, g) in zg.iter().enumerate() {
        for (i, v) in zvals[j].iter_mut().enumerate() {
            if v.is_none() {
                let ob = g.slow.iter().find(|(k, _)| *k == i).map(|(_, o)| o.clone());
                let ob = match ob { Some(o) => o, None => PyString::new(py, std::str::from_utf8(&g.t.text[g.t.pos[i].0 as usize..(g.t.pos[i].0 + g.t.pos[i].1) as usize]).unwrap()).into_any() };
                *v = Some(str_int_mod(&ob, &q)?); // int()'s ValueError, as gmpy2.mpz(z) would raise
            }
        }
    }
    // every gathered text of the slots, parsed in parallel without the GIL
    let texts: Vec<&GText> = slots.iter().flat_map(|row| row.iter()).flat_map(|slot| match slot {
        Slot::Dec(gs) => gs.iter().map(|g| &g.t).collect::<Vec<_>>(),
        Slot::Bin(_) => Vec::new(),
    }).collect();
    let parsed_cells: Vec<Mutex<Vec<Option<u128>>>> = (0..texts.len()).map(|_| Mutex::new(Vec::new())).collect();
    py.detach(|| parallel_for(texts.len(), threads, |i| *parsed_cells[i].lock().unwrap() = parse(texts[i])));
    let mut parsed = parsed_cells.into_iter().map(|m| m.into_inner().unwrap());
    let mut slot_vals: Vec<Vec<Vec<u128>>> = Vec::with_capacity(slots.len());
    for (j, row) in slots.iter().enumerate() {
        let mut out_row = Vec::with_capacity(s_count);
        for (s, slot) in row.iter().enumerate() {
            let malformed = || PyValueError::new_err(("malformed ciphertext", s, j));
            let vals: Vec<u128> = match slot {
                Slot::Bin(b) => {
                    let mut v = vec![0u128; (eta + 1) * n];
                    bin_read(b, qv, ring_logn(&ring), eta, &mut v).ok_or_else(malformed)?;
                    v
                }
                Slot::Dec(gs) => {
                    let mut v = Vec::with_capacity((eta + 1) * n);
                    for g in gs {
                        for (i, p) in parsed.next().unwrap().into_iter().enumerate() {
                            v.push(match p {
                                Some(x) => x,
                                None => {
                                    let ob = g.slow.iter().find(|(k, _)| *k == i).map(|(_, o)| o.clone()).unwrap_or_else(|| {
                                        let (a, l) = g.t.pos[i];
                                        PyString::new(py, std::str::from_utf8(&g.t.text[a as usize..(a + l) as usize]).unwrap()).into_any()
                                    });
                                    str_int_mod(&ob, &q).map_err(|_| malformed())?
                                }
                            });
                        }
                    }
                    v
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
    let big = with_ring!(&*ring, r => {
        let m = &r.m;
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(m.reduce_u128(v))).collect();
        let hats: Vec<Vec<_>> = sk_hats.iter().map(|h| bytes_to_words(h, n)).collect::<PyResult<_>>()?;
        let zw: Vec<Vec<_>> = zvals.iter().map(|z| z.iter().map(|v| m.reduce_u128(v.unwrap())).collect()).collect();
        let sw: Vec<Vec<Vec<_>>> = slot_vals.iter().map(|row| row.iter().map(|v| v.iter().map(|&x| m.reduce_u128(x)).collect()).collect()).collect();
        let dw = m.reduce_u128(d);
        let outs: Vec<Mutex<(Vec<i64>, Vec<u8>, bool)>> = ks.iter().map(|&k| Mutex::new((vec![0; k as usize], vec![0; k as usize], true))).collect();
        py.detach(|| parallel_for(sw.len(), threads, |j| {
            let slot_refs: Vec<&[_]> = sw[j].iter().map(|v| v.as_slice()).collect();
            let hat_refs: Vec<&[_]> = hats.iter().map(|v| v.as_slice()).collect();
            let mut o = outs[j].lock().unwrap();
            let (ref mut vals, ref mut nn, ref mut ok) = *o;
            *ok = rlwe::decrypt_chunk(r, eta, &zw[j], &slot_refs, &ym, &hat_refs, dw, bound_v, vals, nn);
        }));
        let mut big = false;
        for (j, o) in outs.into_iter().enumerate() {
            let (v, nn, ok) = o.into_inner().unwrap();
            big |= !ok;
            values[starts[j]..starts[j] + v.len()].copy_from_slice(&v);
            none[starts[j]..starts[j] + nn.len()].copy_from_slice(&nn);
        }
        big
    });
    if big {
        return Err(fallback()); // |values| beyond int64: let Python decode them
    }
    let data = PyBytes::new(py, &values.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>());
    let mask: Py<PyAny> = if none.iter().any(|&b| b == 1) { PyBytes::new(py, &none).into_any().unbind() } else { py.None() };
    Ok(PyTuple::new(py, [data.into_any().unbind(), mask])?.into_any().unbind())
}

fn ring_logn(r: &AnyRing) -> u32 {
    with_ring!(r, x => x.logn)
}

/// MCFE key terms z'' of several labels as decimal strings.
#[pyfunction]
fn pad_terms(py: Python<'_>, ctx: &Bound<'_, RingCtx>, base: &Bound<'_, PyAny>, materials: &Bound<'_, PyAny>,
             ys: &Bound<'_, PyAny>, eta: usize, threads: usize) -> PyResult<Py<PyList>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let (n, qv) = (ring.n(), ring.q());
    let basev = read_values(base, Some(n), &q, qv, false)?;
    let y_rows: Vec<Vec<u128>> = ys.try_iter()?.map(|y| read_values(&y?, Some(eta), &q, qv, false)).collect::<PyResult<_>>()?;
    let mats: Vec<Vec<Vec<u8>>> = materials.try_iter()?.map(|row| {
        let row = row?;
        let v: Vec<Vec<u8>> = row.try_iter()?.map(|m| m?.cast::<PyBytes>().map(|b| b.as_bytes().to_vec())
            .map_err(|_| PyTypeError::new_err("materials must be bytes"))).collect::<PyResult<_>>()?;
        if v.len() != y_rows.len() {
            return Err(PyValueError::new_err("one material per slot and label"));
        }
        Ok(v)
    }).collect::<PyResult<_>>()?;
    with_ring!(&*ring, r => {
        let m = &r.m;
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(m.reduce_u128(v))).collect();
        let bw: Vec<_> = basev.iter().map(|&v| m.reduce_u128(v)).collect();
        let refs: Vec<Vec<&[u8]>> = mats.iter().map(|row| row.iter().map(|v| v.as_slice()).collect()).collect();
        let outs: Vec<Mutex<Option<DecBuf>>> = (0..refs.len()).map(|_| Mutex::new(None)).collect();
        py.detach(|| rlwe::pad_terms(r, &bw, &refs, &ym, eta, threads, |l, z| *outs[l].lock().unwrap() = Some(DecBuf::new(z))));
        let lists = outs.into_iter().map(|o| o.into_inner().unwrap().unwrap().list(py)).collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, lists)?.unbind())
    })
}

// ------------------------------------------------------ offline / online

enum AnyPre {
    W64(Precomputed<u64>),
    W128(Precomputed<u128>),
}

/// The plaintext-independent part of an MCFE ndarray encryption (one-time).
#[pyclass(module = "pyfe4ai._rust_native")]
struct PrecomputedObj {
    key: Py<EncKeyObj>,
    pre: Mutex<AnyPre>,
}

#[pyfunction]
#[pyo3(signature = (key, pad_materials, sigma2, sigma3, rnd, threads, seed=None))]
#[allow(clippy::too_many_arguments)]
fn precompute_tensor(py: Python<'_>, key: &Bound<'_, EncKeyObj>, pad_materials: &Bound<'_, PyAny>, sigma2: f64,
                     sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, seed: Option<Vec<u8>>)
    -> PyResult<PrecomputedObj> {
    let obj = key.get();
    let mats = materials_of(pad_materials)?.unwrap_or_default();
    let refs: Vec<&[u8]> = mats.iter().map(|v| v.as_slice()).collect();
    let rb: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    let stream = |c: usize, buf: &mut [u8]| drbg.fill(c as u64, buf);
    let pre = match (&*obj.ring, &obj.key) {
        (AnyRing::W64(r), AnyKey::W64(k)) => AnyPre::W64(py.detach(|| rlwe::precompute(r, k, &refs, sigma2, sigma3, randomness(&rb, &stream), threads)).map_err(core_err)?),
        (AnyRing::W128(r), AnyKey::W128(k)) => AnyPre::W128(py.detach(|| rlwe::precompute(r, k, &refs, sigma2, sigma3, randomness(&rb, &stream), threads)).map_err(core_err)?),
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
    };
    let d = pyo3::types::PyDict::new(py);
    d.set_item("chunks", chunks)?;
    d.set_item("used", used)?;
    Ok(d.into_any().unbind())
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
    for f in [
        wrap_pyfunction!(ctx_new, m)?, wrap_pyfunction!(ctx_info, m)?, wrap_pyfunction!(pack, m)?,
        wrap_pyfunction!(unpack, m)?, wrap_pyfunction!(ntt, m)?, wrap_pyfunction!(ntt_mont, m)?,
        wrap_pyfunction!(poly_mul, m)?, wrap_pyfunction!(lincomb, m)?, wrap_pyfunction!(gauss, m)?,
        wrap_pyfunction!(shake256_py, m)?, wrap_pyfunction!(hash_to_zq_py, m)?, wrap_pyfunction!(random_bytes, m)?,
        wrap_pyfunction!(enc_key, m)?, wrap_pyfunction!(encrypt, m)?, wrap_pyfunction!(encrypt_tensor, m)?,
        wrap_pyfunction!(decrypt_tensor, m)?, wrap_pyfunction!(pad_terms, m)?,
        wrap_pyfunction!(precompute_tensor, m)?, wrap_pyfunction!(encrypt_precomputed, m)?,
        wrap_pyfunction!(precomputed_info, m)?, wrap_pyfunction!(dense_dots, m)?, wrap_pyfunction!(build_info, m)?,
        wrap_pyfunction!(drbg_bytes, m)?,
    ] {
        m.add_function(f)?;
    }
    ec::register(m)?;
    Ok(())
}
