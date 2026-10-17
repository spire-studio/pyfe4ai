//! ristretto255 for the */ddh_ristretto255 schemes (curve25519-dalek):
//! points as 32-byte canonical encodings, scalars as 32-byte little-endian
//! integers (reduced mod L by the caller or here), the RFC 9380
//! hash_to_ristretto255 (expand_message_xmd with the core's SHA-512), a bounded discrete
//! log by baby-step giant-step over encodings, and the per-element loops of
//! the MCFE ndarray helpers. Scalar multiplications by secret scalars are the
//! constant-time ones of curve25519-dalek.

use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::{Identity, MultiscalarMul};
use pyfe4ai_core::parallel_for;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};
use pyfe4ai_core::sha512::expand_message_xmd_64;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn point(b: &[u8]) -> PyResult<RistrettoPoint> {
    let arr: [u8; 32] = b.try_into().map_err(|_| PyValueError::new_err("a ristretto255 encoding has 32 bytes"))?;
    CompressedRistretto(arr).decompress().ok_or_else(|| PyValueError::new_err("invalid ristretto255 encoding"))
}

fn scalar(b: &[u8]) -> PyResult<Scalar> {
    let arr: [u8; 32] = b.try_into().map_err(|_| PyValueError::new_err("a scalar has 32 bytes"))?;
    Ok(Scalar::from_bytes_mod_order(arr))
}

fn enc(py: Python<'_>, p: &RistrettoPoint) -> Py<PyBytes> {
    PyBytes::new(py, p.compress().as_bytes()).unbind()
}

fn hash_point(msg: &[u8], dst: &[u8]) -> RistrettoPoint {
    RistrettoPoint::from_uniform_bytes(&expand_message_xmd_64(msg, dst))
}

#[pyfunction]
fn r255_base(py: Python<'_>) -> Py<PyBytes> {
    enc(py, &RISTRETTO_BASEPOINT_POINT)
}

#[pyfunction]
fn r255_check(b: &[u8]) -> bool {
    point(b).is_ok()
}

#[pyfunction]
fn r255_add(py: Python<'_>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    Ok(enc(py, &(point(a)? + point(b)?)))
}

#[pyfunction]
fn r255_sub(py: Python<'_>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    Ok(enc(py, &(point(a)? - point(b)?)))
}

#[pyfunction]
fn r255_neg(py: Python<'_>, a: &[u8]) -> PyResult<Py<PyBytes>> {
    Ok(enc(py, &(-point(a)?)))
}

#[pyfunction]
fn r255_mul(py: Python<'_>, p: &[u8], k: &[u8]) -> PyResult<Py<PyBytes>> {
    Ok(enc(py, &(point(p)? * scalar(k)?)))
}

#[pyfunction]
fn r255_mul_base(py: Python<'_>, k: &[u8]) -> PyResult<Py<PyBytes>> {
    Ok(enc(py, &RistrettoPoint::mul_base(&scalar(k)?)))
}

/// sum_i scalars[i] points[i] (constant time).
#[pyfunction]
fn r255_lincomb(py: Python<'_>, points: Vec<Vec<u8>>, scalars: Vec<Vec<u8>>) -> PyResult<Py<PyBytes>> {
    if points.len() != scalars.len() {
        return Err(PyValueError::new_err("one scalar per point"));
    }
    let ps = points.iter().map(|p| point(p)).collect::<PyResult<Vec<_>>>()?;
    let ks = scalars.iter().map(|k| scalar(k)).collect::<PyResult<Vec<_>>>()?;
    Ok(enc(py, &RistrettoPoint::multiscalar_mul(ks.iter(), ps.iter())))
}

#[pyfunction]
fn r255_hash(py: Python<'_>, msg: &[u8], dst: &[u8]) -> PyResult<Py<PyBytes>> {
    if dst.len() > 255 {
        return Err(PyValueError::new_err("DST longer than 255 bytes"));
    }
    Ok(enc(py, &hash_point(msg, dst)))
}

// ------------------------------------------------------------ discrete log

struct Table {
    m: u64,
    giant: RistrettoPoint,
    baby: HashMap<[u8; 32], u64>,
}

const BABY_FACTOR: u64 = 16;

fn table(bound: u64) -> Arc<Table> {
    static TABLES: OnceLock<Mutex<HashMap<u64, Arc<Table>>>> = OnceLock::new();
    let tables = TABLES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(t) = tables.lock().unwrap().get(&bound) {
        return t.clone();
    }
    let span = 2 * bound + 1;
    let m = span.min(((span as f64).sqrt() as u64 + 1) * BABY_FACTOR);
    let mut baby = HashMap::with_capacity(m as usize);
    let mut p = RistrettoPoint::identity();
    for j in 0..m {
        baby.insert(p.compress().to_bytes(), j);
        p += RISTRETTO_BASEPOINT_POINT;
    }
    let t = Arc::new(Table { m, giant: -(RISTRETTO_BASEPOINT_POINT * Scalar::from(m)), baby });
    tables.lock().unwrap().insert(bound, t.clone());
    t
}

fn dlog(p: RistrettoPoint, bound: u64, t: &Table) -> Option<i64> {
    let mut gamma = p + RISTRETTO_BASEPOINT_POINT * Scalar::from(bound);
    for i in 0..(2 * bound) / t.m + 2 {
        if let Some(&j) = t.baby.get(gamma.compress().as_bytes()) {
            let x = (i * t.m + j) as i64 - bound as i64;
            return if x.unsigned_abs() <= bound { Some(x) } else { None };
        }
        gamma += t.giant;
    }
    None
}

/// The x with x G = p and |x| <= bound, or None.
#[pyfunction]
fn r255_dlog(py: Python<'_>, p: &[u8], bound: u64) -> PyResult<Option<i64>> {
    let pt = point(p)?;
    Ok(py.detach(|| dlog(pt, bound, &table(bound))))
}

// ------------------------------------------------- MCFE ndarray, per element

/// c_i = x_i G + s0 H(0 || msg_i) + s1 H(1 || msg_i) for every element, as hex strings.
#[pyfunction]
fn r255_mcfe_encrypt(py: Python<'_>, msgs: Vec<Vec<u8>>, dst: &[u8], xs: Vec<i64>, s0: &[u8], s1: &[u8],
                     threads: usize) -> PyResult<Py<PyList>> {
    if msgs.len() != xs.len() {
        return Err(PyValueError::new_err("one label per value"));
    }
    let (k0, k1) = (scalar(s0)?, scalar(s1)?);
    let outs: Vec<Mutex<String>> = (0..xs.len()).map(|_| Mutex::new(String::new())).collect();
    py.detach(|| parallel_for(xs.len(), threads, |i| {
        let mut m0 = vec![0u8];
        m0.extend_from_slice(&msgs[i]);
        let mut m1 = vec![1u8];
        m1.extend_from_slice(&msgs[i]);
        let x = Scalar::from(xs[i].unsigned_abs());
        let xg = if xs[i] < 0 { -x } else { x };
        let c = RistrettoPoint::multiscalar_mul(
            [xg, k0, k1].iter(),
            [RISTRETTO_BASEPOINT_POINT, hash_point(&m0, dst), hash_point(&m1, dst)].iter(),
        );
        *outs[i].lock().unwrap() = c.compress().as_bytes().iter().map(|b| format!("{:02x}", b)).collect();
    }));
    Ok(PyList::new(py, outs.into_iter().map(|s| PyString::new(py, &s.into_inner().unwrap())))?.unbind())
}

/// For every element i: sum_c y_c c_{c,i} - d0 H(0 || msg_i) - d1 H(1 || msg_i) and its
/// bounded dlog. cts[c][i]: hex encodings; ys[c]: small weights. Returns (int64 bytes,
/// None mask | None); raises ValueError(("malformed ciphertext", c, i)).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn r255_mcfe_decrypt(py: Python<'_>, cts: Vec<Vec<String>>, ys: Vec<i64>, d0: &[u8], d1: &[u8], msgs: Vec<Vec<u8>>,
                     dst: &[u8], bound: u64, threads: usize) -> PyResult<Py<PyAny>> {
    if cts.len() != ys.len() || cts.iter().any(|c| c.len() != msgs.len()) {
        return Err(PyValueError::new_err("inconsistent decryption inputs"));
    }
    let mut pts = Vec::with_capacity(cts.len());
    for (c, col) in cts.iter().enumerate() {
        let mut v = Vec::with_capacity(col.len());
        for (i, h) in col.iter().enumerate() {
            let bytes: Option<Vec<u8>> = (0..h.len() / 2)
                .map(|k| u8::from_str_radix(h.get(2 * k..2 * k + 2)?, 16).ok())
                .collect();
            let p = bytes.filter(|b| b.len() == 32 && h.len() == 64).and_then(|b| point(&b).ok());
            v.push(p.ok_or_else(|| PyValueError::new_err(("malformed ciphertext", c, i)))?);
        }
        pts.push(v);
    }
    let (k0, k1) = (scalar(d0)?, scalar(d1)?);
    let yk: Vec<Scalar> = ys.iter().map(|&y| if y < 0 { -Scalar::from(y.unsigned_abs()) } else { Scalar::from(y as u64) }).collect();
    let t = table(bound);
    let n = msgs.len();
    let outs: Vec<Mutex<Option<i64>>> = (0..n).map(|_| Mutex::new(None)).collect();
    py.detach(|| parallel_for(n, threads, |i| {
        let mut m0 = vec![0u8];
        m0.extend_from_slice(&msgs[i]);
        let mut m1 = vec![1u8];
        m1.extend_from_slice(&msgs[i]);
        let mut scalars: Vec<Scalar> = yk.clone();
        let mut points: Vec<RistrettoPoint> = pts.iter().map(|col| col[i]).collect();
        scalars.push(-k0);
        scalars.push(-k1);
        points.push(hash_point(&m0, dst));
        points.push(hash_point(&m1, dst));
        let v = RistrettoPoint::multiscalar_mul(scalars.iter(), points.iter());
        *outs[i].lock().unwrap() = dlog(v, bound, &t);
    }));
    let vals: Vec<Option<i64>> = outs.into_iter().map(|m| m.into_inner().unwrap()).collect();
    let data: Vec<u8> = vals.iter().flat_map(|v| v.unwrap_or(0).to_ne_bytes()).collect();
    let mask: Py<PyAny> = if vals.iter().any(|v| v.is_none()) {
        PyBytes::new(py, &vals.iter().map(|v| v.is_none() as u8).collect::<Vec<u8>>()).into_any().unbind()
    } else {
        py.None()
    };
    Ok(PyTuple::new(py, [PyBytes::new(py, &data).into_any().unbind(), mask])?.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(r255_base, m)?, wrap_pyfunction!(r255_check, m)?, wrap_pyfunction!(r255_add, m)?,
        wrap_pyfunction!(r255_sub, m)?, wrap_pyfunction!(r255_neg, m)?, wrap_pyfunction!(r255_mul, m)?,
        wrap_pyfunction!(r255_mul_base, m)?, wrap_pyfunction!(r255_lincomb, m)?, wrap_pyfunction!(r255_hash, m)?,
        wrap_pyfunction!(r255_dlog, m)?, wrap_pyfunction!(r255_mcfe_encrypt, m)?, wrap_pyfunction!(r255_mcfe_decrypt, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
