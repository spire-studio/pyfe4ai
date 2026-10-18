//! The lattice functions of the extension for an RNS modulus q = q1 q2
//! (pyfe4ai-core `rns`): a context, encryption keys and precomputations of
//! their own classes, which the functions of lib.rs dispatch to. Values
//! cross the boundary as for a prime modulus (Python ints, decimal strings,
//! format-1 bytes, packed 128-bit words in [0, q)); only the NTT-domain keys
//! (`ntt_mont`) hold residues (2n 64-bit words).

use super::{
    bytes_to_words, chunk_list, core_err, dec_list, delta_mod, exact_err, exact_fn, int64_buffer, materials_of,
    py_index, randomness, read_rows, read_values, words_to_bytes, CallDrbg, ChunkOut, DecBuf, FacctArg, TensorRing,
};
use pyfe4ai_core::codec::{bin_size, bin_wb, bin_write};
use pyfe4ai_core::facct;
use pyfe4ai_core::gauss::gauss_bytes;
use pyfe4ai_core::keccak::{hash_to_zq_mode, HashMode};
use pyfe4ai_core::rlwe::{FacctNoise, Noise};
use pyfe4ai_core::rns::{self, RnsKey, RnsPrecomputed, RnsRing};
use pyfe4ai_core::{rlwe, Modulus, Word};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::sync::{Arc, Mutex};

/// Z_q[x] / (x^n + 1) for q = q1 q2 with one NTT per prime (rns_ctx_new).
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
pub struct RnsCtx {
    ring: Arc<RnsRing>,
    q: Py<PyAny>,
}

#[pyfunction]
pub fn rns_ctx_new(py: Python<'_>, q: &Bound<'_, PyAny>, n: usize, primes: Vec<u64>, roots: Vec<u64>) -> PyResult<RnsCtx> {
    if primes.len() != 2 || roots.len() != 2 {
        return Err(PyValueError::new_err("an RNS context has two primes and two roots"));
    }
    let qv: u128 = py_index(q)?.extract().map_err(|_| PyValueError::new_err("q must be below 2^124"))?;
    let ring = py
        .detach(|| RnsRing::new(qv, n, [primes[0], primes[1]], [roots[0], roots[1]]))
        .map_err(PyValueError::new_err)?;
    Ok(RnsCtx { ring: Arc::new(ring), q: py_index(q)?.unbind() })
}

pub fn ctx_info(py: Python<'_>, ctx: &Bound<'_, RnsCtx>) -> PyResult<Py<PyAny>> {
    let r = &ctx.get().ring;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("word_bits", 128)?;
    d.set_item("n", r.n)?;
    d.set_item("q_bits", r.qbits)?;
    d.set_item("rns", vec![r.r[0].q, r.r[1].q])?;
    Ok(d.into_any().unbind())
}

pub fn pack(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let c = ctx.get();
    Ok(words_to_bytes(py, &read_values(values, None, c.q.bind(py), &c.ring.m, false)?))
}

pub fn unpack(py: Python<'_>, data: &[u8], as_digits: bool) -> PyResult<Py<PyList>> {
    if data.len() % 16 != 0 {
        return Err(PyValueError::new_err("length is not a multiple of the word size"));
    }
    let w: Vec<u128> = data.chunks_exact(16).map(u128::read_le).collect();
    if as_digits {
        return Ok(dec_list(py, &w)?.unbind());
    }
    Ok(PyList::new(py, w)?.unbind())
}

pub fn ntt_mont(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, data: &[u8]) -> PyResult<Py<PyBytes>> {
    let r = &ctx.get().ring;
    let x: Vec<u128> = bytes_to_words(data, r.n)?;
    Ok(words_to_bytes(py, &r.ntt_mont_std(&x)))
}

/// The NTT mod q with the CRT root (ring_lwe_utils.ntt_forward / ntt_inverse):
/// one NTT per prime with the root's residue, joined (standard values in [0, q)).
pub fn ntt(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, data: &[u8], inverse: bool) -> PyResult<Py<PyBytes>> {
    let r = &ctx.get().ring;
    let x: Vec<u128> = bytes_to_words(data, r.n)?;
    let (mut res, mut t) = (vec![0u64; 2 * r.n], vec![0u64; 2 * r.n]);
    r.split(&x, &mut res);
    if inverse { r.inverse(&res, &mut t) } else { r.forward(&res, &mut t) }
    let mut out = vec![0u128; r.n];
    r.join(&t, &mut out);
    Ok(words_to_bytes(py, &out))
}

/// The negacyclic product mod q (ring_lwe_utils.poly_mul).
pub fn poly_mul(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    let r = &ctx.get().ring;
    let (x, y): (Vec<u128>, Vec<u128>) = (bytes_to_words(a, r.n)?, bytes_to_words(b, r.n)?);
    Ok(words_to_bytes(py, &r.poly_mul(&x, &y)))
}

/// crypto_utils.hash_to_zq_vector of the material mod q in `mode`, packed
/// 128-bit words (the native label pad), wiped after packing.
pub fn hash_to_zq(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, material: &[u8], count: usize, mode: HashMode) -> Py<PyBytes> {
    let r = &ctx.get().ring;
    let mut out = vec![0u128; count];
    py.detach(|| hash_to_zq_mode(&r.m, material, mode, &mut out));
    let bytes = words_to_bytes(py, &out);
    rlwe::wipe_words(&mut out);
    bytes
}

pub fn lincomb(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, rows: &Bound<'_, PyAny>, coeffs: &Bound<'_, PyAny>)
    -> PyResult<Py<PyList>> {
    let c = ctx.get();
    let (r, m) = (&c.ring, &c.ring.m);
    let rows: Vec<Vec<u8>> = rows.try_iter()?.map(|r| r?.extract::<Vec<u8>>()).collect::<PyResult<_>>()?;
    let ys = read_values(coeffs, None, c.q.bind(py), m, false)?;
    if rows.len() != ys.len() {
        return Err(PyValueError::new_err("one coefficient per row"));
    }
    // the rows may be secret (a client's pads u): every copy is wiped
    let mut rows = rows;
    let out = (|| {
        let mut acc = vec![0u128; r.n];
        for (row, &y) in rows.iter().zip(ys.iter()) {
            let mut v: Vec<u128> = bytes_to_words(row, r.n)?;
            let ym = m.to_mont(y);
            for i in 0..r.n {
                acc[i] = m.add(acc[i], m.mul(v[i], ym));
            }
            rlwe::wipe_words(&mut v);
        }
        let out = dec_list(py, &acc).map(|l| l.unbind());
        rlwe::wipe_words(&mut acc);
        out
    })();
    for row in rows.iter_mut() {
        rlwe::wipe_bytes(row);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn pad_terms(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, base: &Bound<'_, PyAny>, materials: &Bound<'_, PyAny>,
                 ys: &Bound<'_, PyAny>, eta: usize, mode: HashMode, threads: usize) -> PyResult<Py<PyList>> {
    let c = ctx.get();
    let (r, m, q) = (&*c.ring, &c.ring.m, c.q.bind(py));
    let basev = read_values(base, Some(r.n), q, m, false)?;
    let y_rows = read_rows(ys, Some(eta), q, m)?;
    let mats: Vec<Vec<Vec<u8>>> = materials.try_iter()?.map(|row| {
        let row = row?;
        let v: Vec<Vec<u8>> = row.try_iter()?.map(|m| m?.cast::<PyBytes>().map(|b| b.as_bytes().to_vec())
            .map_err(|_| PyTypeError::new_err("materials must be bytes"))).collect::<PyResult<_>>()?;
        if v.len() != y_rows.len() {
            return Err(PyValueError::new_err("one material per slot and label"));
        }
        Ok(v)
    }).collect::<PyResult<_>>()?;
    let ym: Vec<u128> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
    let refs: Vec<Vec<&[u8]>> = mats.iter().map(|row| row.iter().map(|v| v.as_slice()).collect()).collect();
    let outs: Vec<Mutex<Option<DecBuf>>> = (0..refs.len()).map(|_| Mutex::new(None)).collect();
    py.detach(|| rns::pad_terms(r, &basev, &refs, &ym, eta, mode, threads, |l, z| *outs[l].lock().unwrap() = Some(DecBuf::new(z))));
    let lists = outs.into_iter().map(|o| o.into_inner().unwrap().unwrap().list(py)).collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, lists)?.unbind())
}

impl TensorRing<u128> for RnsRing {
    type Hat = u64;
    fn modulus(&self) -> &Modulus<u128> {
        &self.m
    }
    fn dims(&self) -> (usize, u32) {
        (self.n, self.logn)
    }
    fn read_hat(&self, b: &[u8]) -> PyResult<Vec<u64>> {
        bytes_to_words(b, 2 * self.n)
    }
    fn decrypt_chunk(&self, eta: usize, z: &[u128], slots: &[&[u128]], ym: &[u128], hats: &[&[u64]], delta: u128,
                     bound: i128, out: &mut [i64], none: &mut [u8]) -> bool {
        rns::decrypt_chunk(self, eta, z, slots, ym, hats, delta, bound, out, none)
    }
}

pub fn ring_of(ctx: &Bound<'_, RnsCtx>) -> (Arc<RnsRing>, Py<PyAny>) {
    let c = ctx.get();
    (c.ring.clone(), c.q.clone_ref(ctx.py()))
}

// --------------------------------------------------------- encryption key

/// A slot's encryption key for an RNS modulus (rns::RnsKey).
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
pub struct RnsKeyObj {
    ring: Arc<RnsRing>,
    q: Py<PyAny>,
    key: RnsKey,
}

pub fn enc_key(py: Python<'_>, ctx: &Bound<'_, RnsCtx>, a: &Bound<'_, PyAny>, pk: &Bound<'_, PyAny>,
               u: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>, mode: HashMode) -> PyResult<RnsKeyObj> {
    let c = ctx.get();
    let (ring, q) = (c.ring.clone(), c.q.bind(py));
    let (n, m) = (ring.n, &ring.m);
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
    let d = delta_mod(delta, q, m)?;
    let mut key = RnsKey::new(&ring, &av, &pk_rows, u_rows.as_deref(), d);
    key.hash_mode = mode;
    Ok(RnsKeyObj { ring, q: q.clone().unbind(), key })
}

fn chunk_out(r: &RnsRing, binary: bool, c0: &[u128], c1: &[u128]) -> ChunkOut {
    if binary {
        let mut out = Vec::with_capacity(bin_size(1, r.n, bin_wb(r.qbits)));
        bin_write(r.qbits, r.logn, 1, c0, c1, &mut out);
        ChunkOut::Bin(out)
    } else {
        ChunkOut::Dec(DecBuf::new(c0), DecBuf::new(c1))
    }
}

/// encrypt (MIFERingLWE._encrypt_rows) for an RNS key.
#[allow(clippy::too_many_arguments)]
pub fn encrypt(py: Python<'_>, key: &Bound<'_, RnsKeyObj>, w_rows: &Bound<'_, PyAny>, sigma2: f64, sigma3: f64,
               rnd: &Bound<'_, PyAny>, binary: bool, seed: Option<Vec<u8>>, facct: FacctArg) -> PyResult<Py<PyAny>> {
    let obj = key.get();
    let (r, k) = (&*obj.ring, &obj.key);
    let n = r.n;
    let rows = read_rows(w_rows, Some(n), obj.q.bind(py), &r.m)?;
    let eta = k.eta;
    if rows.len() != eta {
        return Err(PyValueError::new_err("one plaintext row per pk row"));
    }
    let mut w = vec![0u64; eta * 2 * n];
    for (row, dst) in rows.iter().zip(w.chunks_exact_mut(2 * n)) {
        r.split(row, dst);
    }
    let (mut c0, mut c1) = (vec![0u128; eta * n], vec![0u128; n]);
    if facct.is_some() {
        // as the prime path: getrandbits from the given bytes or stream 0 of the call's DRBG
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
                py.detach(|| rns::encrypt_chunk_facct(r, k, &w, nz, &mut bits, &mut c0, &mut c1));
            }
        });
        if bits.failed {
            rlwe::wipe_words(&mut w);
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
        py.detach(|| rns::encrypt_chunk(r, k, &w, sigma2, sigma3, &rb, &mut c0, &mut c1));
        rlwe::wipe_bytes(&mut rb);
    }
    rlwe::wipe_words(&mut w); // x + u: the pad u is secret
    if binary {
        let mut out = Vec::new();
        bin_write(r.qbits, r.logn, eta, &c0, &c1, &mut out);
        return Ok(PyBytes::new(py, &out).into_any().unbind());
    }
    let rows_out = PyList::new(py, c0.chunks(n).map(|row| dec_list(py, row)).collect::<PyResult<Vec<_>>>()?)?;
    Ok(PyTuple::new(py, [rows_out.into_any(), dec_list(py, &c1)?.into_any()])?.into_any().unbind())
}

/// encrypt_tensor for an RNS key.
#[allow(clippy::too_many_arguments)]
pub fn encrypt_tensor(py: Python<'_>, key: &Bound<'_, RnsKeyObj>, x: &Bound<'_, PyAny>, pad_materials: &Bound<'_, PyAny>,
                      sigma2: f64, sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, binary: bool,
                      seed: Option<Vec<u8>>, facct: FacctArg) -> PyResult<Py<PyList>> {
    let obj = key.get();
    let (r, k) = (&*obj.ring, &obj.key);
    let xs = int64_buffer(x)?;
    let mats = materials_of(pad_materials)?;
    let rb: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    let stream = |c: usize, off: u64, buf: &mut [u8]| drbg.fill_at(c as u64, off, buf);
    let n = r.n;
    let chunks = if xs.is_empty() { 0 } else { (xs.len() + n - 1) / n };
    let outs: Vec<Mutex<Option<ChunkOut>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
    let refs: Option<Vec<&[u8]>> = mats.as_ref().map(|m| m.iter().map(|v| v.as_slice()).collect());
    with_noise!(py, facct, noise, err, {
        py.detach(|| rns::encrypt_tensor(r, k, &xs, refs.as_deref(), sigma2, sigma3, &noise, randomness(&rb, &stream),
            threads, |j, c0, c1| *outs[j].lock().unwrap() = Some(chunk_out(r, binary, c0, c1))))
            .map_err(core_err)?
    });
    chunk_list(py, outs.into_iter().map(|m| m.into_inner().unwrap()).collect(), n, xs.len())
}

// ------------------------------------------------------ offline / online

/// The plaintext-independent part of an MCFE ndarray encryption for an RNS key (one-time).
#[pyclass(module = "pyfe4ai._rust_native")]
pub struct RnsPreObj {
    key: Py<RnsKeyObj>,
    pre: Mutex<RnsPrecomputed>,
    /// The process that made it: after a fork the child's copy is unused too,
    /// and using it there would encrypt another input with the same r and noise.
    pid: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn precompute_tensor(py: Python<'_>, key: &Bound<'_, RnsKeyObj>, pad_materials: &Bound<'_, PyAny>, sigma2: f64,
                         sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, seed: Option<Vec<u8>>, facct: FacctArg)
    -> PyResult<RnsPreObj> {
    let obj = key.get();
    let mats = materials_of(pad_materials)?.unwrap_or_default();
    let refs: Vec<&[u8]> = mats.iter().map(|v| v.as_slice()).collect();
    let rb: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    let stream = |c: usize, off: u64, buf: &mut [u8]| drbg.fill_at(c as u64, off, buf);
    let pre = with_noise!(py, facct, noise, err, {
        py.detach(|| rns::precompute(&obj.ring, &obj.key, &refs, sigma2, sigma3, &noise, randomness(&rb, &stream),
            threads)).map_err(core_err)?
    });
    Ok(RnsPreObj { key: key.clone().unbind(), pre: Mutex::new(pre), pid: std::process::id() })
}

/// The precomputation's lock, waited for detached from the interpreter: the call
/// holding it runs its kernel with the GIL released, so a second thread
/// blocking on it with the GIL attached would deadlock the process.
fn lock_pre<'a>(py: Python<'_>, p: &'a RnsPreObj) -> PyResult<std::sync::MutexGuard<'a, RnsPrecomputed>> {
    p.pre.lock_py_attached(py).map_err(|_| PyValueError::new_err("precomputation poisoned by a failed call"))
}

pub fn encrypt_precomputed(py: Python<'_>, key: &Bound<'_, RnsKeyObj>, pre: &Bound<'_, PyAny>, first: usize,
                           x: &Bound<'_, PyAny>, threads: usize, binary: bool) -> PyResult<Py<PyList>> {
    let p = pre.cast::<RnsPreObj>().map_err(|_| PyValueError::new_err("precomputed with another key"))?.borrow();
    if !p.key.bind(py).is(key) {
        return Err(PyValueError::new_err("precomputed with another key"));
    }
    if p.pid != std::process::id() {
        return Err(PyValueError::new_err("precomputed in another process (a fork): one-time, use it there only"));
    }
    let xs = int64_buffer(x)?;
    let obj = key.get();
    let (r, k) = (&*obj.ring, &obj.key);
    let mut guard = lock_pre(py, &p)?;
    let pc: &mut RnsPrecomputed = &mut guard;
    let n = r.n;
    let chunks = (xs.len() + n - 1) / n;
    let outs: Vec<Mutex<Option<ChunkOut>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
    py.detach(|| rns::encrypt_precomputed(r, k, pc, first, &xs, threads,
        |j, c0, c1| *outs[j].lock().unwrap() = Some(chunk_out(r, binary, c0, c1)))).map_err(core_err)?;
    chunk_list(py, outs.into_iter().map(|m| m.into_inner().unwrap()).collect(), n, xs.len())
}

pub fn precomputed_info(py: Python<'_>, pre: &Bound<'_, RnsPreObj>) -> PyResult<Py<PyAny>> {
    let p = pre.borrow();
    let g = lock_pre(py, &p)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("chunks", g.chunks())?;
    d.set_item("used", g.used.iter().filter(|&&u| u).count())?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RnsCtx>()?;
    m.add_class::<RnsKeyObj>()?;
    m.add_class::<RnsPreObj>()?;
    m.add_function(wrap_pyfunction!(rns_ctx_new, m)?)?;
    Ok(())
}
