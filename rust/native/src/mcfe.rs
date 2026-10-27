//! Ring-LWE MCFE and dMCFE: the encryption of a layer (encrypt_tensor), the key
//! terms of labels (pad_terms), and the offline / online split of the layer
//! encryption (precompute_tensor, encrypt_precomputed).

use crate::drbg::{randomness, CallDrbg};
use crate::hashing::hash_mode;
use crate::mife::{with_key, AnyKey, EncKeyObj};
use crate::ring::{get_ring, RingCtx};
use crate::rns;
use crate::sampling::{exact_err, exact_fn, with_noise, FacctArg};
use crate::secret::{secret_materials, wipe_materials};
use crate::threshold;
use crate::values::{int64_buffer, materials_of, read_rows, read_values, DecBuf};
use pyfe4ai_core::codec::{bin_size, bin_wb, bin_write};
use pyfe4ai_core::rlwe::{self, FacctNoise, Noise, Precomputed};
use pyfe4ai_core::{with_ring, AnyRing, Ring, Word, U256};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::sync::Mutex;

pub(crate) fn core_err(e: rlwe::Error) -> PyErr {
    match e {
        rlwe::Error::Input(m) => PyValueError::new_err(m),
        rlwe::Error::Random(e) => PyOSError::new_err(e.to_string()),
    }
}

/// Per-chunk outputs of the tensor encryptions, formatted in the workers.
pub(crate) enum ChunkOut {
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

pub(crate) fn chunk_list(py: Python<'_>, outs: Vec<Option<ChunkOut>>, n: usize, len: usize) -> PyResult<Py<PyList>> {
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

/// MCFE ndarray encryption of one layer (eta = 1); see pyfe4ai-core rlwe::encrypt_tensor.
#[pyfunction]
#[pyo3(signature = (key, x, pad_materials, sigma2, sigma3, rnd, threads, binary=false, seed=None, facct=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encrypt_tensor(py: Python<'_>, key: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, pad_materials: &Bound<'_, PyAny>,
                  sigma2: f64, sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, binary: bool,
                  seed: Option<Vec<u8>>, facct: FacctArg) -> PyResult<Py<PyList>> {
    if let Ok(k) = key.cast::<rns::RnsKeyObj>() {
        return rns::encrypt_tensor(py, k, x, pad_materials, sigma2, sigma3, rnd, threads, binary, seed, facct);
    }
    let obj = key.cast::<EncKeyObj>()?.get();
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

/// MCFE key terms z'' of several labels as decimal strings, or with `binary`
/// (RingCtx only) as bytes of ceil(bits(q) / 8)-byte words.
#[pyfunction]
#[pyo3(signature = (ctx, base, materials, ys, eta, threads, hash_mode=0, binary=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pad_terms(py: Python<'_>, ctx: &Bound<'_, PyAny>, base: &Bound<'_, PyAny>, materials: &Bound<'_, PyAny>,
             ys: &Bound<'_, PyAny>, eta: usize, threads: usize, hash_mode: u8, binary: bool) -> PyResult<Py<PyList>> {
    let mode = self::hash_mode(hash_mode)?;
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        if binary {
            return Err(PyValueError::new_err("binary key terms need a ring context"));
        }
        return rns::pad_terms(py, c, base, materials, ys, eta, mode, threads);
    }
    let ctx = ctx.cast::<RingCtx>()?;
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let n = ring.n();
    with_ring!(&*ring, r => {
        let m = &r.m;
        // base (sum_r y_r u_r) and the materials (u', dMCFE's K_ij) are secret: wiped after use
        let mut basev = read_values(base, Some(n), &q, m, false)?;
        let out = (|| {
            let y_rows = read_rows(ys, Some(eta), &q, m)?;
            let mut mats = secret_materials(materials, y_rows.len())?;
            let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
            if binary {
                let wb = threshold::value_bytes_of(m);
                let outs: Vec<Mutex<Vec<u8>>> = (0..mats.len()).map(|_| Mutex::new(Vec::new())).collect();
                {
                    let refs: Vec<Vec<&[u8]>> = mats.iter().map(|row| row.iter().map(|v| v.as_slice()).collect()).collect();
                    py.detach(|| rlwe::pad_terms(r, &basev, &refs, &ym, eta, mode, threads, |l, z| {
                        let mut buf = Vec::with_capacity(z.len() * wb);
                        pyfe4ai_core::threshold::write_values(z, wb, &mut buf);
                        *outs[l].lock().unwrap() = buf;
                    }));
                }
                wipe_materials(&mut mats);
                let blobs: Vec<Py<PyBytes>> = outs.into_iter().map(|o| {
                    let mut buf = o.into_inner().unwrap();
                    let out = PyBytes::new(py, &buf).unbind();
                    rlwe::wipe_bytes(&mut buf); // the key terms are secret
                    out
                }).collect();
                return Ok(PyList::new(py, blobs)?.unbind());
            }
            let outs: Vec<Mutex<Option<DecBuf>>> = (0..mats.len()).map(|_| Mutex::new(None)).collect();
            {
                let refs: Vec<Vec<&[u8]>> = mats.iter().map(|row| row.iter().map(|v| v.as_slice()).collect()).collect();
                py.detach(|| rlwe::pad_terms(r, &basev, &refs, &ym, eta, mode, threads,
                    |l, z| *outs[l].lock().unwrap() = Some(DecBuf::new(z))));
            }
            wipe_materials(&mut mats);
            let lists = outs.into_iter().map(|o| o.into_inner().unwrap().unwrap().list(py)).collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, lists)?.unbind())
        })();
        rlwe::wipe_words(&mut basev);
        out
    })
}

enum AnyPre {
    W64(Precomputed<u64>),
    W128(Precomputed<u128>),
    W256(Precomputed<U256>),
}

/// The plaintext-independent part of an MCFE ndarray encryption (one-time;
/// usable only in the process that made it, not in a forked child).
#[pyclass(module = "pyfe4ai._rust_native")]
pub(crate) struct PrecomputedObj {
    key: Py<EncKeyObj>,
    pre: Mutex<AnyPre>,
    pid: u32,
}

/// The precomputation's lock, waited for detached from the interpreter: the call
/// holding it runs its kernel with the GIL released, so a second thread
/// blocking on it with the GIL attached would deadlock the process.
fn lock_pre<'a>(py: Python<'_>, p: &'a PrecomputedObj) -> PyResult<std::sync::MutexGuard<'a, AnyPre>> {
    if p.pid != std::process::id() {
        return Err(PyValueError::new_err("precomputation made in another process"));
    }
    p.pre.lock_py_attached(py).map_err(|_| PyValueError::new_err("precomputation poisoned by a failed call"))
}

#[pyfunction]
#[pyo3(signature = (key, pad_materials, sigma2, sigma3, rnd, threads, seed=None, facct=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn precompute_tensor(py: Python<'_>, key: &Bound<'_, PyAny>, pad_materials: &Bound<'_, PyAny>, sigma2: f64,
                     sigma3: f64, rnd: &Bound<'_, PyAny>, threads: usize, seed: Option<Vec<u8>>, facct: FacctArg)
    -> PyResult<Py<PyAny>> {
    if let Ok(k) = key.cast::<rns::RnsKeyObj>() {
        let pre = rns::precompute_tensor(py, k, pad_materials, sigma2, sigma3, rnd, threads, seed, facct)?;
        return Ok(Py::new(py, pre)?.into_any());
    }
    let key = key.cast::<EncKeyObj>()?;
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
    Ok(Py::new(py, PrecomputedObj { key: key.clone().unbind(), pre: Mutex::new(pre), pid: std::process::id() })?.into_any())
}

#[pyfunction]
#[pyo3(signature = (key, pre, first, x, threads, binary=false))]
pub(crate) fn encrypt_precomputed(py: Python<'_>, key: &Bound<'_, PyAny>, pre: &Bound<'_, PyAny>, first: usize,
                       x: &Bound<'_, PyAny>, threads: usize, binary: bool) -> PyResult<Py<PyList>> {
    if let Ok(k) = key.cast::<rns::RnsKeyObj>() {
        return rns::encrypt_precomputed(py, k, pre, first, x, threads, binary);
    }
    let key = key.cast::<EncKeyObj>()?;
    let p = pre.cast::<PrecomputedObj>().map_err(|_| PyValueError::new_err("precomputed with another key"))?.borrow();
    if !p.key.bind(py).is(key) {
        return Err(PyValueError::new_err("precomputed with another key"));
    }
    let xs = int64_buffer(x)?;
    let obj = key.get();
    let mut guard = lock_pre(py, &p)?;
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
pub(crate) fn precomputed_info(py: Python<'_>, pre: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(r) = pre.cast::<rns::RnsPreObj>() {
        return rns::precomputed_info(py, r);
    }
    let p = pre.cast::<PrecomputedObj>()?.borrow();
    let g = lock_pre(py, &p)?;
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
