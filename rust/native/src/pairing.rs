//! BLS12-381 for the pairing-based schemes (arkworks), with the element
//! interface the schemes use from charm-crypto: G1, G2 and GT written
//! multiplicatively (`*` the group operation, `**` exponentiation, `/`), ZR the
//! scalar field (`+ - * / **`, `int()`), `==` and `hash()` on all of them.
//! pyfe4ai.primitives.pairing.bls12_381 wraps this into a PairingGroup.
//!
//! Elements are kept in memory as curve points (projective) / field elements;
//! serialization is the arkworks compressed form, deserialization checks
//! the encoding and subgroup membership. Random elements are g^r for scalars
//! from 64 OS-random bytes reduced mod r. Exponentiations and products of
//! powers (`bls_msm`, `bls_pow_many`) use the methods of `msm.rs`. Batches
//! (`bls_deserialize_many`, `bls_serialize_many`, `bls_msm_many`) and
//! `bls_multi_pair` take a number of threads, and give the same result on any
//! number of them. `bls_prepare_g2` stores the lines of fixed G2 points, and
//! `bls_multi_pair_prepared` takes one product of pairings per row of G1 points
//! over those lines, with the same results as `bls_multi_pair`. Not
//! constant-time in general.

use crate::dlog::{self, TableCache};
use crate::msm;
use ark_bls12_381::{Bls12_381, Config, Fq12, Fq12Config, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bls12::{Bls12Config, G2Prepared};
use ark_ec::pairing::{MillerLoopOutput, Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{BigInteger, BitIteratorBE, CyclotomicMultSubgroup, Field, Fp12Config, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use pyfe4ai_core::{osrand, parallel_for};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyInt};
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

/// charm's type codes.
pub const ZR: u8 = 0;
pub const G1: u8 = 1;
pub const G2: u8 = 2;
pub const GT: u8 = 3;

type Gt = PairingOutput<Bls12_381>;

/// From this many items on a batch uses threads: below, starting them costs about as much as the work.
const PARALLEL_MIN: usize = 16;

/// [f(i) for i in 0..count], on up to `threads` threads.
fn map_threads<T: Send, F: Fn(usize) -> T + Sync>(count: usize, threads: usize, f: F) -> Vec<T> {
    let outs: Vec<Mutex<Option<T>>> = (0..count).map(|_| Mutex::new(None)).collect();
    parallel_for(count, threads, |i| {
        *outs[i].lock().unwrap() = Some(f(i));
    });
    outs.into_iter().map(|out| out.into_inner().unwrap().expect("parallel_for visits every index")).collect()
}

/// [f(i) for i in 0..count], on up to `threads` threads for a batch of at least `PARALLEL_MIN` items.
fn map_parallel<T: Send, F: Fn(usize) -> T + Sync>(count: usize, threads: usize, f: F) -> Vec<T> {
    map_threads(count, if count < PARALLEL_MIN { 1 } else { threads }, f)
}

// GT (576 bytes) dominates; boxing it would cost an allocation per element
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Val {
    Zr(Fr),
    G1(G1Projective),
    G2(G2Projective),
    Gt(Gt),
}

/// An element of ZR, G1, G2 or GT of BLS12-381.
#[pyclass(frozen, module = "pyfe4ai._rust_native", name = "Bls12381Element")]
#[derive(Clone)]
pub struct Elem {
    v: Val,
}

fn fr_from_int(ob: &Bound<'_, PyAny>) -> PyResult<Fr> {
    if let Ok(e) = ob.cast::<Elem>() {
        return match e.get().v {
            Val::Zr(x) => Ok(x),
            _ => Err(PyTypeError::new_err("exponent must be an integer or a ZR element")),
        };
    }
    if let Ok(i) = ob.cast_exact::<PyInt>() {
        // an int of 64 bits, without raising
        let mut overflow: std::os::raw::c_int = 0;
        let v = unsafe { pyo3::ffi::PyLong_AsLongLongAndOverflow(i.as_ptr(), &mut overflow) };
        if overflow == 0 {
            return Ok(Fr::from(v));
        }
    }
    // operator.index(ob) % r
    let py = ob.py();
    let v = unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyNumber_Index(ob.as_ptr())) }?;
    let v = v.rem(cached_order(py)?)?;
    let bytes = v.call_method1("to_bytes", (32, "little"))?;
    Ok(Fr::from_le_bytes_mod_order(bytes.cast::<PyBytes>()?.as_bytes()))
}

/// The Fr of every item of a sequence of ints / ZR elements.
fn frs_from_seq(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Fr>> {
    seq.try_iter()?.map(|ob| fr_from_int(&ob?)).collect()
}

fn cached_order(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static ORDER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Ok(ORDER.get_or_try_init(py, || order_int(py).map(Bound::unbind))?.bind(py))
}

fn order_int(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let bytes = Fr::MODULUS.to_bytes_le();
    let int = py.import("builtins")?.getattr("int")?;
    int.call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))
}

fn fr_to_int<'py>(py: Python<'py>, x: &Fr) -> PyResult<Bound<'py, PyAny>> {
    let bytes = x.into_bigint().to_bytes_le();
    py.import("builtins")?.getattr("int")?.call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))
}

fn random_fr() -> PyResult<Fr> {
    let mut b = [0u8; 64];
    osrand::fill(&mut b).map_err(|e| PyOSError::new_err(e.to_string()))?;
    let x = Fr::from_le_bytes_mod_order(&b);
    pyfe4ai_core::rlwe::wipe_bytes(&mut b);
    Ok(x)
}

fn elem(v: Val) -> Elem {
    Elem { v }
}

/// The per-process key of [`gt_key`].
fn gt_hash_key() -> &'static RandomState {
    static KEY: OnceLock<RandomState> = OnceLock::new();
    KEY.get_or_init(RandomState::new)
}

/// Build the lazy statics of this module now: the module calls this at import, with the GIL held and
/// before any fork, so no thread ever waits for another to build one (a wait parks the thread, which
/// a forked child on macOS does not survive once that thread parked in the parent; see
/// `pyfe4ai_core::thread_limit`).
pub(crate) fn warm_statics(py: Python<'_>) -> PyResult<()> {
    cached_order(py)?;
    gt_generator();
    gt_hash_key();
    Ok(())
}

/// e(g1, g2) for the fixed generators.
fn gt_generator() -> Gt {
    static GT_GEN: OnceLock<Gt> = OnceLock::new();
    *GT_GEN.get_or_init(|| Bls12_381::pairing(G1Projective::generator(), G2Projective::generator()))
}

/// The power of a group element (not ZR) by e.
fn group_pow(v: &Val, e: &Fr) -> Val {
    match v {
        Val::G1(p) => Val::G1(msm::curve_msm(&[*p], &[*e])),
        Val::G2(p) => Val::G2(msm::g2_pow(p, e)),
        Val::Gt(t) => Val::Gt(PairingOutput(msm::gt_msm(&[t.0], &[*e]))),
        Val::Zr(_) => unreachable!("ZR is not a group here"),
    }
}

/// GT from its encoding: the Fq12 coefficients are checked canonical, and
/// the membership test of msm.rs replaces arkworks' f^r == 1.
fn gt_deserialize(data: &[u8]) -> Option<Gt> {
    let f = Fq12::deserialize_with_mode(data, Compress::Yes, Validate::Yes).ok()?;
    msm::gt_is_member(&f).then_some(PairingOutput(f))
}

/// Bytes of the compressed encodings of ZR, G1, G2 and GT.
const ENCODED_LEN: [usize; 4] = [32, 48, 96, 576];

/// The element of type t of exactly its compressed encoding (no trailing bytes).
fn deserialize_val(t: u8, data: &[u8]) -> PyResult<Val> {
    let bad = |_| PyValueError::new_err("invalid BLS12-381 encoding");
    if t <= GT && data.len() != ENCODED_LEN[t as usize] {
        return Err(PyValueError::new_err("invalid BLS12-381 encoding"));
    }
    Ok(match t {
        ZR => Val::Zr(Fr::deserialize_compressed(data).map_err(bad)?),
        G1 => Val::G1(ark_bls12_381::G1Affine::deserialize_compressed(data).map_err(bad)?.into()),
        G2 => Val::G2(ark_bls12_381::G2Affine::deserialize_compressed(data).map_err(bad)?.into()),
        GT => Val::Gt(gt_deserialize(data).ok_or_else(|| PyValueError::new_err("invalid BLS12-381 encoding"))?),
        _ => return Err(PyValueError::new_err("unknown group type")),
    })
}

fn kind(v: &Val) -> u8 {
    match v {
        Val::Zr(_) => ZR,
        Val::G1(_) => G1,
        Val::G2(_) => G2,
        Val::Gt(_) => GT,
    }
}

fn other_elem(ob: &Bound<'_, PyAny>) -> Option<Val> {
    ob.cast::<Elem>().ok().map(|e| e.get().v)
}

fn serialize_val(v: &Val) -> Vec<u8> {
    let mut out = Vec::new();
    match v {
        Val::Zr(x) => x.serialize_compressed(&mut out),
        Val::G1(p) => p.into_affine().serialize_compressed(&mut out),
        Val::G2(p) => p.into_affine().serialize_compressed(&mut out),
        Val::Gt(t) => t.serialize_compressed(&mut out),
    }
    .expect("serialization into a vector");
    out
}

#[pymethods]
impl Elem {
    /// charm's type code: 0 ZR, 1 G1, 2 G2, 3 GT.
    #[getter(r#type)]
    fn type_(&self) -> u8 {
        kind(&self.v)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        let o = match other_elem(other) {
            Some(o) => o,
            None => match self.v {
                Val::Zr(a) => return Ok(elem(Val::Zr(a * fr_from_int(other)?))),
                _ => return Err(PyTypeError::new_err("group elements multiply with elements of their group")),
            },
        };
        Ok(elem(match (self.v, o) {
            (Val::Zr(a), Val::Zr(b)) => Val::Zr(a * b),
            (Val::G1(a), Val::G1(b)) => Val::G1(a + b),
            (Val::G2(a), Val::G2(b)) => Val::G2(a + b),
            (Val::Gt(a), Val::Gt(b)) => Val::Gt(a + b),
            _ => return Err(PyTypeError::new_err("elements of different groups")),
        }))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        let o = match other_elem(other) {
            Some(o) => o,
            None => match self.v {
                Val::Zr(_) => Val::Zr(fr_from_int(other)?),
                _ => return Err(PyTypeError::new_err("group elements divide by elements of their group")),
            },
        };
        Ok(elem(match (self.v, o) {
            (Val::Zr(a), Val::Zr(b)) => {
                Val::Zr(a * b.inverse().ok_or_else(|| PyZeroDivisionError::new_err("division by zero in ZR"))?)
            }
            (Val::G1(a), Val::G1(b)) => Val::G1(a - b),
            (Val::G2(a), Val::G2(b)) => Val::G2(a - b),
            (Val::Gt(a), Val::Gt(b)) => Val::Gt(a - b),
            _ => return Err(PyTypeError::new_err("elements of different groups")),
        }))
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        match self.v {
            Val::Zr(b) => {
                let a = fr_from_int(other)?;
                Ok(elem(Val::Zr(a * b.inverse().ok_or_else(|| PyZeroDivisionError::new_err("division by zero in ZR"))?)))
            }
            // 1 / g: the inverse, as in charm
            _ if fr_from_int(other).map(|a| a == Fr::one()).unwrap_or(false) => Ok(elem(match self.v {
                Val::G1(p) => Val::G1(-p),
                Val::G2(p) => Val::G2(-p),
                Val::Gt(t) => Val::Gt(-t),
                Val::Zr(_) => unreachable!(),
            })),
            _ => Err(PyTypeError::new_err("unsupported division")),
        }
    }

    fn __pow__(&self, exponent: &Bound<'_, PyAny>, modulo: &Bound<'_, PyAny>) -> PyResult<Elem> {
        if !modulo.is_none() {
            return Err(PyTypeError::new_err("pow() with a modulus is not supported"));
        }
        let a = match self.v {
            Val::Zr(a) => a,
            v => return Ok(elem(group_pow(&v, &fr_from_int(exponent)?))),
        };
        if let Some(v) = other_elem(exponent) {
            // a ZR exponent: its representative in [0, r)
            return match v {
                Val::Zr(k) => Ok(elem(Val::Zr(a.pow(k.into_bigint())))),
                _ => Err(PyTypeError::new_err("exponent must be an integer or a ZR element")),
            };
        }
        // an integer k: a^k = a^(k mod (r - 1)) for a != 0 (Fermat), k < 0 included;
        // reducing k mod r would be wrong for |k| >= r
        let py = exponent.py();
        let k = unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyNumber_Index(exponent.as_ptr())) }?;
        if a.is_zero() {
            return if k.lt(0)? {
                Err(PyZeroDivisionError::new_err("0 ** negative in ZR"))
            } else {
                Ok(elem(Val::Zr(if k.eq(0)? { Fr::one() } else { Fr::zero() })))
            };
        }
        if let Ok(v) = k.extract::<i64>() {
            // |k| < r - 1: the field power, inverted for k < 0
            let p = a.pow([v.unsigned_abs()]);
            return Ok(elem(Val::Zr(if v < 0 {
                p.inverse().ok_or_else(|| PyZeroDivisionError::new_err("0 ** negative in ZR"))?
            } else {
                p
            })));
        }
        let e = fr_from_int(&k.rem(cached_order(py)?.sub(1)?)?)?; // in [0, r - 1): exact
        Ok(elem(Val::Zr(a.pow(e.into_bigint()))))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        match self.v {
            Val::Zr(a) => Ok(elem(Val::Zr(a + match other_elem(other) {
                Some(Val::Zr(b)) => b,
                Some(_) => return Err(PyTypeError::new_err("ZR + group element")),
                None => fr_from_int(other)?,
            }))),
            _ => Err(PyTypeError::new_err("group elements combine with *")),
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        match self.v {
            Val::Zr(a) => Ok(elem(Val::Zr(a - match other_elem(other) {
                Some(Val::Zr(b)) => b,
                Some(_) => return Err(PyTypeError::new_err("ZR - group element")),
                None => fr_from_int(other)?,
            }))),
            _ => Err(PyTypeError::new_err("group elements combine with /")),
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Elem> {
        match self.v {
            Val::Zr(b) => Ok(elem(Val::Zr(fr_from_int(other)? - b))),
            _ => Err(PyTypeError::new_err("group elements combine with /")),
        }
    }

    fn __neg__(&self) -> PyResult<Elem> {
        Ok(elem(match self.v {
            Val::Zr(a) => Val::Zr(-a),
            Val::G1(p) => Val::G1(-p),
            Val::G2(p) => Val::G2(-p),
            Val::Gt(t) => Val::Gt(-t),
        }))
    }

    fn __int__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.v {
            Val::Zr(x) => fr_to_int(py, &x),
            _ => Err(PyTypeError::new_err("only ZR elements convert to int")),
        }
    }

    fn __index__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.__int__(py)
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let eq = match other_elem(other) {
            Some(o) => self.v == o,
            // ZR == int compares residues mod r
            None => match self.v {
                Val::Zr(a) => fr_from_int(other).map(|b| a == b).unwrap_or(false),
                _ => false,
            },
        };
        match op {
            CompareOp::Eq => Ok(eq.into_pyobject(py)?.to_owned().into_any().unbind()),
            CompareOp::Ne => Ok((!eq).into_pyobject(py)?.to_owned().into_any().unbind()),
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        kind(&self.v).hash(&mut h);
        serialize_val(&self.v).hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        let name = ["ZR", "G1", "G2", "GT"][kind(&self.v) as usize];
        let b = serialize_val(&self.v);
        format!("<BLS12-381 {} {}>", name, b.iter().take(8).map(|x| format!("{:02x}", x)).collect::<String>())
    }

    /// The canonical compressed encoding (arkworks), without the type.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &serialize_val(&self.v))
    }
}

/// A random element of ZR (0), G1 (1), G2 (2) or GT (3).
#[pyfunction]
fn bls_random(t: u8) -> PyResult<Elem> {
    let x = random_fr()?;
    Ok(elem(match t {
        ZR => Val::Zr(x),
        G1 => group_pow(&Val::G1(G1Projective::generator()), &x),
        G2 => group_pow(&Val::G2(G2Projective::generator()), &x),
        GT => group_pow(&Val::Gt(gt_generator()), &x),
        _ => return Err(PyValueError::new_err("unknown group type")),
    }))
}

/// The identity of G1 / G2 / GT, or ZR(value).
#[pyfunction]
#[pyo3(signature = (t, value=None))]
fn bls_init(t: u8, value: Option<&Bound<'_, PyAny>>) -> PyResult<Elem> {
    Ok(elem(match t {
        ZR => Val::Zr(match value {
            Some(v) => fr_from_int(v)?,
            None => Fr::zero(),
        }),
        G1 => Val::G1(G1Projective::zero()),
        G2 => Val::G2(G2Projective::zero()),
        GT => Val::Gt(Gt::zero()),
        _ => return Err(PyValueError::new_err("unknown group type")),
    }))
}

/// The fixed generators of G1 / G2 (and e(g1, g2) of GT).
#[pyfunction]
fn bls_generator(t: u8) -> PyResult<Elem> {
    Ok(elem(match t {
        G1 => Val::G1(G1Projective::generator()),
        G2 => Val::G2(G2Projective::generator()),
        GT => Val::Gt(gt_generator()),
        _ => return Err(PyValueError::new_err("ZR has no generator")),
    }))
}

#[pyfunction]
fn bls_order(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    order_int(py)
}

/// Element of type t from its canonical compressed encoding (checked).
#[pyfunction]
fn bls_deserialize(t: u8, data: &[u8]) -> PyResult<Elem> {
    deserialize_val(t, data).map(elem)
}

/// Elements from (type, encoding) pairs, each checked as by bls_deserialize;
/// equal pairs are decoded once. Raises ValueError at the first invalid one.
#[pyfunction]
#[pyo3(signature = (items, threads=1))]
fn bls_deserialize_many(py: Python<'_>, items: Vec<(u8, Vec<u8>)>, threads: usize) -> PyResult<Vec<Elem>> {
    let (decoded, slots) = py.detach(|| {
        let mut first: HashMap<(u8, &[u8]), usize> = HashMap::new();
        let mut distinct: Vec<(u8, &[u8])> = Vec::new();
        let slots: Vec<usize> = items
            .iter()
            .map(|(t, data)| {
                *first.entry((*t, data.as_slice())).or_insert_with(|| {
                    distinct.push((*t, data.as_slice()));
                    distinct.len() - 1
                })
            })
            .collect();
        (map_parallel(distinct.len(), threads, |i| deserialize_val(distinct[i].0, distinct[i].1)), slots)
    });
    slots
        .into_iter()
        .map(|slot| match &decoded[slot] {
            Ok(v) => Ok(elem(*v)),
            Err(e) => Err(e.clone_ref(py)),
        })
        .collect()
}

/// [serialize_val(v) for v in vals], with one inversion per group instead of one per point.
fn serialize_vals(vals: &[Val], threads: usize) -> Vec<Vec<u8>> {
    let g1: Vec<_> = vals.iter().filter_map(|v| if let Val::G1(p) = v { Some(*p) } else { None }).collect();
    let g2: Vec<_> = vals.iter().filter_map(|v| if let Val::G2(p) = v { Some(*p) } else { None }).collect();
    let (g1, g2) = (G1Projective::normalize_batch(&g1), G2Projective::normalize_batch(&g2));
    // each point's place in its group, in the order of vals
    let (mut in_g1, mut in_g2) = (0..g1.len(), 0..g2.len());
    let places: Vec<usize> = vals
        .iter()
        .map(|v| match v {
            Val::G1(_) => in_g1.next().expect("one place per G1 point"),
            Val::G2(_) => in_g2.next().expect("one place per G2 point"),
            _ => 0,
        })
        .collect();
    map_parallel(vals.len(), threads, |i| {
        let mut out = Vec::new();
        match &vals[i] {
            Val::G1(_) => g1[places[i]].serialize_compressed(&mut out),
            Val::G2(_) => g2[places[i]].serialize_compressed(&mut out),
            other => return serialize_val(other),
        }
        .expect("serialization into a vector");
        out
    })
}

/// [x.to_bytes() for x in elements].
#[pyfunction]
#[pyo3(signature = (elements, threads=1))]
fn bls_serialize_many<'py>(py: Python<'py>, elements: Vec<Bound<'py, Elem>>, threads: usize)
    -> Vec<Bound<'py, PyBytes>> {
    let vals: Vec<Val> = elements.iter().map(|e| e.get().v).collect();
    let raw = py.detach(|| serialize_vals(&vals, threads));
    raw.iter().map(|data| PyBytes::new(py, data)).collect()
}

/// The bases of a product of powers: elements of one group (G1, G2 or GT).
fn group_bases(bases: &[Bound<'_, Elem>]) -> PyResult<Vec<Val>> {
    let vals: Vec<Val> = bases.iter().map(|b| b.get().v).collect();
    match vals.first() {
        None => Err(PyValueError::new_err("a product of powers needs at least one base")),
        Some(Val::Zr(_)) => Err(PyTypeError::new_err("bases must be G1, G2 or GT elements")),
        Some(v0) if vals.iter().any(|v| kind(v) != kind(v0)) => {
            Err(PyTypeError::new_err("bases of a product of powers must be of one group"))
        }
        _ => Ok(vals),
    }
}

/// The bases and the scalars of a product of powers, of equal length.
fn msm_terms(bases: &[Bound<'_, Elem>], scalars: &Bound<'_, PyAny>) -> PyResult<(Vec<Val>, Vec<Fr>)> {
    let vals = group_bases(bases)?;
    let ks = frs_from_seq(scalars)?;
    if ks.len() != vals.len() {
        return Err(PyValueError::new_err("msm takes as many scalars as bases"));
    }
    Ok((vals, ks))
}

/// prod_i vals[i] ** ks[i] for the terms of `msm_terms`.
fn msm_vals(vals: &[Val], ks: &[Fr]) -> Val {
    // group_bases checked that every base is of the group of the first
    match vals[0] {
        Val::G1(_) => {
            let ps: Vec<_> = vals.iter().filter_map(|v| if let Val::G1(p) = v { Some(*p) } else { None }).collect();
            Val::G1(msm::curve_msm(&ps, &ks))
        }
        Val::G2(p) if vals.len() == 1 => Val::G2(msm::g2_pow(&p, &ks[0])),
        Val::G2(_) => {
            let ps: Vec<_> = vals.iter().filter_map(|v| if let Val::G2(p) = v { Some(*p) } else { None }).collect();
            Val::G2(msm::curve_msm(&ps, &ks))
        }
        Val::Gt(_) => {
            let fs: Vec<_> = vals.iter().filter_map(|v| if let Val::Gt(t) = v { Some(t.0) } else { None }).collect();
            Val::Gt(PairingOutput(msm::gt_msm(&fs, &ks)))
        }
        Val::Zr(_) => unreachable!("group_bases rejects ZR"),
    }
}

/// prod_i bases[i] ** scalars[i] for bases of one group (G1, G2 or GT) and
/// scalars ints or ZR elements: one multi-scalar multiplication.
#[pyfunction]
fn bls_msm(py: Python<'_>, bases: Vec<Bound<'_, Elem>>, scalars: &Bound<'_, PyAny>) -> PyResult<Elem> {
    let (vals, ks) = msm_terms(&bases, scalars)?;
    Ok(elem(py.detach(|| msm_vals(&vals, &ks))))
}

/// [bls_msm(b, k) for b, k in zip(bases, scalars)].
#[pyfunction]
#[pyo3(signature = (bases, scalars, threads=1))]
fn bls_msm_many(py: Python<'_>, bases: Vec<Vec<Bound<'_, Elem>>>, scalars: Vec<Bound<'_, PyAny>>, threads: usize)
    -> PyResult<Vec<Elem>> {
    if bases.len() != scalars.len() {
        return Err(PyValueError::new_err("msm_many takes as many rows of scalars as rows of bases"));
    }
    let rows = bases.iter().zip(&scalars).map(|(b, k)| msm_terms(b, k)).collect::<PyResult<Vec<_>>>()?;
    let out = py.detach(|| map_parallel(rows.len(), threads, |i| msm_vals(&rows[i].0, &rows[i].1)));
    Ok(out.into_iter().map(elem).collect())
}

/// [base ** k for k in scalars] for base in G1, G2 or GT (one table for the base).
#[pyfunction]
fn bls_pow_many(py: Python<'_>, base: &Bound<'_, Elem>, scalars: &Bound<'_, PyAny>) -> PyResult<Vec<Elem>> {
    let v = base.get().v;
    if let Val::Zr(_) = v {
        return Err(PyTypeError::new_err("the base must be a G1, G2 or GT element"));
    }
    let ks = frs_from_seq(scalars)?;
    let out: Vec<Val> = py.detach(|| match v {
        Val::G1(p) => msm::curve_pow_many(&p, &ks).into_iter().map(Val::G1).collect(),
        Val::G2(p) => msm::curve_pow_many(&p, &ks).into_iter().map(Val::G2).collect(),
        Val::Gt(t) => msm::gt_pow_many(&t.0, &ks).into_iter().map(|f| Val::Gt(PairingOutput(f))).collect(),
        Val::Zr(_) => unreachable!("checked above"),
    });
    Ok(out.into_iter().map(elem).collect())
}

/// e(a, b) for a in G1 and b in G2 (either order).
#[pyfunction]
fn bls_pair(py: Python<'_>, a: &Bound<'_, Elem>, b: &Bound<'_, Elem>) -> PyResult<Elem> {
    let (p, q) = match (a.get().v, b.get().v) {
        (Val::G1(p), Val::G2(q)) | (Val::G2(q), Val::G1(p)) => (p, q),
        _ => return Err(PyValueError::new_err("pair() takes a G1 and a G2 element")),
    };
    Ok(elem(Val::Gt(py.detach(|| Bls12_381::pairing(p, q)))))
}

/// prod_i e(ps_i, qs_i): the Miller loops of a chunk of the pairs on each thread, their product, and one final
/// exponentiation.
fn multi_pairing(ps: &[G1Projective], qs: &[G2Projective], threads: usize) -> Gt {
    let (ps, qs) = (G1Projective::normalize_batch(ps), G2Projective::normalize_batch(qs));
    if threads < 2 || ps.len() < PARALLEL_MIN {
        return Bls12_381::multi_pairing(ps, qs);
    }
    let size = ps.len().div_ceil(threads);
    let loops = map_threads(ps.len().div_ceil(size), threads, |i| {
        let (p, q) = (ps.chunks(size).nth(i).expect("a chunk"), qs.chunks(size).nth(i).expect("a chunk"));
        Bls12_381::multi_miller_loop(p.iter().copied(), q.iter().copied()).0
    });
    let product = loops.into_iter().fold(Fq12::one(), |acc, part| acc * part);
    Bls12_381::final_exponentiation(MillerLoopOutput(product)).expect("the Miller loop of curve points is nonzero")
}

/// prod_i e(a_i, b_i): one Miller loop product and one final exponentiation.
/// Each pair is a G1 and a G2 element in either order, as for pair().
#[pyfunction]
#[pyo3(signature = (a, b, threads=1))]
fn bls_multi_pair(py: Python<'_>, a: Vec<Bound<'_, Elem>>, b: Vec<Bound<'_, Elem>>, threads: usize) -> PyResult<Elem> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("multi_pair takes as many left as right elements"));
    }
    let mut ps = Vec::with_capacity(a.len());
    let mut qs = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(b.iter()) {
        match (x.get().v, y.get().v) {
            (Val::G1(p), Val::G2(q)) | (Val::G2(q), Val::G1(p)) => {
                ps.push(p);
                qs.push(q);
            }
            _ => return Err(PyValueError::new_err("multi_pair() takes pairs of a G1 and a G2 element")),
        }
    }
    Ok(elem(Val::Gt(py.detach(|| multi_pairing(&ps, &qs, threads)))))
}

// ------------------------------------------- products of pairings over prepared G2 points

/// f *= 1 + c1 v + c4 v w: ark-ff's `mul_by_014` with c0 = 1.
#[inline]
fn mul_by_unit_line(f: &mut Fq12, c1: &Fq2, c4: &Fq2) {
    let mut aa = f.c0;
    let mut t = f.c0;
    t.mul_by_1(c1);
    aa += &t;
    let mut bb = f.c1;
    bb.mul_by_1(c4);
    let mut o = *c1;
    o += c4;
    f.c1 += &f.c0;
    let mut t = f.c1;
    t.mul_by_1(&o);
    f.c1 += &t;
    f.c1 -= &aa;
    f.c1 -= &bb;
    f.c0 = bb;
    Fq12Config::mul_fp6_by_nonresidue_in_place(&mut f.c0);
    f.c0 += &aa;
}

/// The lines of the Miller loop of a G2 point, in the order of the loop.
enum Lines {
    /// The identity of G2 has none, and contributes nothing to a product.
    Identity,
    /// (c0, c1, c2) of every line, as in arkworks' `G2Prepared`.
    Full(Vec<(Fq2, Fq2, Fq2)>),
    /// (c1 / c0, c2 / c0): each line differs from the full one by a constant of Fq2, which the final exponentiation
    /// removes.
    Unit(Vec<(Fq2, Fq2)>),
}

impl Lines {
    /// The lines of q: `Unit` if `normalise` is set and no line has c0 = 0, a point with such a line stays `Full`.
    fn of(q: G2Affine, normalise: bool) -> Lines {
        let prepared = G2Prepared::<Config>::from(q);
        if prepared.infinity {
            return Lines::Identity;
        }
        let lines = prepared.ell_coeffs;
        if !normalise || lines.iter().any(|c| c.0.is_zero()) {
            return Lines::Full(lines);
        }
        let mut inverses: Vec<Fq2> = lines.iter().map(|c| c.0).collect();
        ark_ff::batch_inversion(&mut inverses);
        Lines::Unit(lines.iter().zip(&inverses).map(|(c, i)| (c.1 * i, c.2 * i)).collect())
    }

    /// f *= line number `at`, evaluated at p, which is not the identity.
    #[inline]
    fn multiply(&self, f: &mut Fq12, at: usize, p: &G1Affine) {
        match self {
            Lines::Identity => {}
            Lines::Full(lines) => {
                let (c0, mut c1, mut c2) = lines[at];
                c1.mul_assign_by_fp(&p.x);
                c2.mul_assign_by_fp(&p.y);
                f.mul_by_014(&c0, &c1, &c2);
            }
            Lines::Unit(lines) => {
                let (mut c1, mut c2) = lines[at];
                c1.mul_assign_by_fp(&p.x);
                c2.mul_assign_by_fp(&p.y);
                mul_by_unit_line(f, &c1, &c2);
            }
        }
    }
}

/// prod_i e(ps[i], q_i) for the G2 points q_i of `lines`: one Miller loop over the pairs and one final exponentiation.
/// The identity of G1 is left out, as in arkworks' `multi_miller_loop`: its coordinates (0, 0) are not a curve point.
fn prepared_product(ps: &[G1Affine], lines: &[Lines]) -> Gt {
    let pairs: Vec<(&G1Affine, &Lines)> = ps.iter().zip(lines).filter(|(p, _)| !p.is_zero()).collect();
    let mut f = Fq12::one();
    let mut at = 0;
    let mut multiply = |f: &mut Fq12| {
        for (p, q) in &pairs {
            q.multiply(f, at, p);
        }
        at += 1;
    };
    for bit in BitIteratorBE::without_leading_zeros(<Config as Bls12Config>::X).skip(1) {
        f.square_in_place();
        multiply(&mut f);
        if bit {
            multiply(&mut f);
        }
    }
    f.cyclotomic_inverse_in_place(); // x is negative
    Bls12_381::final_exponentiation(MillerLoopOutput(f)).expect("the Miller loop of curve points is nonzero")
}

/// The lines of the Miller loops of some G2 points, in their order. A prepared point is as sensitive as the key
/// element it comes from, not a public value: there is no `__repr__` to print it.
#[pyclass(frozen, module = "pyfe4ai._rust_native", name = "PreparedG2")]
pub struct PreparedG2 {
    lines: Vec<Lines>,
}

/// The lines of `points` (at least one, all in G2) for `bls_multi_pair_prepared`. `normalise` has one flag per point,
/// whether to store its lines scaled to c0 = 1; without it, all of them are.
#[pyfunction]
#[pyo3(signature = (points, threads=1, normalise=None))]
fn bls_prepare_g2(py: Python<'_>, points: Vec<Bound<'_, Elem>>, threads: usize, normalise: Option<Vec<bool>>)
    -> PyResult<PreparedG2> {
    if points.is_empty() {
        return Err(PyValueError::new_err("prepare_g2 needs at least one point"));
    }
    let normalise = normalise.unwrap_or_else(|| vec![true; points.len()]);
    if normalise.len() != points.len() {
        return Err(PyValueError::new_err(format!(
            "prepare_g2: normalise has {} flags for {} points", normalise.len(), points.len()
        )));
    }
    let mut qs = Vec::with_capacity(points.len());
    for (at, point) in points.iter().enumerate() {
        match point.get().v {
            Val::G2(q) => qs.push(q),
            _ => return Err(PyValueError::new_err(format!("prepare_g2: position {at}: expected a G2 element"))),
        }
    }
    let lines = py.detach(|| {
        let qs = G2Projective::normalize_batch(&qs);
        map_parallel(qs.len(), threads, |i| Lines::of(qs[i], normalise[i]))
    });
    Ok(PreparedG2 { lines })
}

/// [prod_i e(row[i], q_i) for row in rows] for the G2 points q_i of `prepared`: one G1 element per point in every row,
/// one final exponentiation per row. The rows are normalised in one batch, and take up to `threads` threads however
/// few they are: a row costs at least a final exponentiation, far more than starting a thread.
#[pyfunction]
#[pyo3(signature = (rows, prepared, threads=1))]
fn bls_multi_pair_prepared(py: Python<'_>, rows: Vec<Vec<Bound<'_, Elem>>>, prepared: &Bound<'_, PreparedG2>,
                           threads: usize) -> PyResult<Vec<Elem>> {
    let lines = prepared.get().lines.as_slice();
    // the lengths first: the room for the elements is reserved from them, and a malformed input of many rows must
    // not reach that
    for (r, row) in rows.iter().enumerate() {
        if row.len() != lines.len() {
            return Err(PyValueError::new_err(format!(
                "multi_pair_prepared: row {r} has {} elements, expected {}", row.len(), lines.len()
            )));
        }
    }
    let mut ps = Vec::with_capacity(rows.len() * lines.len());
    for (r, row) in rows.iter().enumerate() {
        for (at, x) in row.iter().enumerate() {
            match x.get().v {
                Val::G1(p) => ps.push(p),
                _ => return Err(PyValueError::new_err(format!(
                    "multi_pair_prepared: row {r}, position {at}: expected a G1 element"
                ))),
            }
        }
    }
    let count = rows.len();
    let out = py.detach(|| {
        let ps = G1Projective::normalize_batch(&ps);
        map_threads(count, threads, |r| prepared_product(&ps[r * lines.len()..(r + 1) * lines.len()], lines))
    });
    Ok(out.into_iter().map(|t| elem(Val::Gt(t))).collect())
}

// ----------------------------------------------------------- dlog in GT

struct GtTable {
    m: u64,
    giant: Gt,
    baby: HashMap<u128, u64>,
}

/// Bytes of one baby step (key and index).
const GT_ENTRY: usize = std::mem::size_of::<(u128, u64)>();

/// The table key of a GT element: 128 bits of SipHash-1-3 over its Fq12
/// coordinates (two inputs, domain-separated) under one random key per process
/// (std's RandomState). Cheaper than hashing its 576-byte encoding; a false match
/// among the at most 2^49 lookups of a dlog has probability below 2^-79 and
/// cannot be aimed at, the key being unknown.
fn gt_key(t: &Gt) -> u128 {
    let key = gt_hash_key();
    let half = |domain: u8| {
        let mut h = key.build_hasher();
        domain.hash(&mut h);
        t.hash(&mut h);
        h.finish() as u128
    };
    (half(0) << 64) | half(1)
}

fn build_gt_table(base: Gt, m: u64) -> GtTable {
    let mut baby = HashMap::with_capacity(m as usize);
    let mut cur = Gt::zero();
    for j in 0..m {
        baby.insert(gt_key(&cur), j);
        cur += base;
    }
    GtTable { m, giant: -(base * Fr::from(m)), baby }
}

/// The cached tables, per (base, size): at most 16 tables / 128 MB (see dlog.rs),
/// least recently used evicted.
static GT_TABLES: TableCache<(u128, u64), GtTable> = TableCache::new(dlog::CACHE_TABLES, dlog::CACHE_BYTES);

fn gt_table(base: Gt, bound: u64) -> Arc<GtTable> {
    let m = dlog::baby_steps(bound, true, GT_ENTRY);
    GT_TABLES.get_or_build((gt_key(&base), m), dlog::table_bytes(m, GT_ENTRY), || build_gt_table(base, m))
}

/// (tables, bytes) of the GT dlog table cache (diagnostics, tests).
#[pyfunction]
fn bls_gt_dlog_cache() -> (usize, usize) {
    GT_TABLES.stats()
}

fn gt_search(t: &GtTable, g: Gt, h: Gt, bound: u64) -> Option<i64> {
    let mut gamma = h + g * Fr::from(bound);
    for i in 0..(2 * bound) / t.m + 2 {
        if let Some(&j) = t.baby.get(&gt_key(&gamma)) {
            let x = (i * t.m + j) as i64 - bound as i64;
            return if x.unsigned_abs() <= bound { Some(x) } else { None };
        }
        gamma += t.giant;
    }
    None
}

/// The x with base^x = target and |x| <= bound (at most 2^48), or None. With
/// `cache` the baby-step table is kept for later calls with this base and bound;
/// without (a base used once, such as SIFE FH-IPE's e(k1, c1)), a table of about
/// sqrt(2 bound + 1) steps is built for this call only.
#[pyfunction]
#[pyo3(signature = (base, target, bound, cache=true))]
fn bls_gt_dlog(py: Python<'_>, base: &Bound<'_, Elem>, target: &Bound<'_, Elem>, bound: u64, cache: bool)
    -> PyResult<Option<i64>> {
    let (g, h) = match (base.get().v, target.get().v) {
        (Val::Gt(g), Val::Gt(h)) => (g, h),
        _ => return Err(PyTypeError::new_err("dlog in GT takes GT elements")),
    };
    if bound > dlog::MAX_BOUND {
        return Err(PyValueError::new_err("dlog bound above 2^48"));
    }
    Ok(py.detach(|| {
        if cache {
            gt_search(&gt_table(g, bound), g, h, bound)
        } else {
            gt_search(&build_gt_table(g, dlog::baby_steps(bound, false, GT_ENTRY)), g, h, bound)
        }
    }))
}

// ------------------------------------------------------------ hash to curve

/// RFC 9380 hash_to_curve for BLS12381G1_XMD:SHA-256_SSWU_RO_ (t = G1) and
/// BLS12381G2_XMD:SHA-256_SSWU_RO_ (t = G2): hash_to_field to two elements
/// (L = 64 bytes per base-field coefficient), the simplified SWU map on the
/// isogenous curve and the isogeny (arkworks' WBMap), and cofactor clearing.
fn hash_to_curve(t: u8, msg: &[u8], dst: &[u8]) -> PyResult<Val> {
    use ark_bls12_381::{g1, g2, Fq, Fq2};
    use ark_ec::hashing::curve_maps::wb::WBMap;
    use ark_ec::hashing::map_to_curve_hasher::MapToCurve;
    use ark_ec::AffineRepr;
    use pyfe4ai_core::sha256::expand_message_xmd;

    let err = |e: ark_ec::hashing::HashToCurveError| PyValueError::new_err(e.to_string());
    let fq = |b: &[u8]| Fq::from_be_bytes_mod_order(b);
    match t {
        G1 => {
            let u = expand_message_xmd(msg, dst, 2 * 64);
            let q0 = <WBMap<g1::Config> as MapToCurve<G1Projective>>::map_to_curve(fq(&u[..64])).map_err(err)?;
            let q1 = <WBMap<g1::Config> as MapToCurve<G1Projective>>::map_to_curve(fq(&u[64..])).map_err(err)?;
            Ok(Val::G1((q0 + q1).into_affine().clear_cofactor().into_group()))
        }
        G2 => {
            let u = expand_message_xmd(msg, dst, 2 * 2 * 64);
            let fq2 = |b: &[u8]| Fq2::new(fq(&b[..64]), fq(&b[64..128]));
            let q0 = <WBMap<g2::Config> as MapToCurve<G2Projective>>::map_to_curve(fq2(&u[..128])).map_err(err)?;
            let q1 = <WBMap<g2::Config> as MapToCurve<G2Projective>>::map_to_curve(fq2(&u[128..])).map_err(err)?;
            Ok(Val::G2((q0 + q1).into_affine().clear_cofactor().into_group()))
        }
        _ => Err(PyValueError::new_err("hashing into ZR or GT is not supported")),
    }
}

/// H(msg) in G1 or G2 (RFC 9380, SSWU, random oracle) with domain separation tag `dst`.
#[pyfunction]
fn bls_hash(t: u8, msg: &[u8], dst: &[u8]) -> PyResult<Elem> {
    if dst.len() > 255 {
        return Err(PyValueError::new_err("DST longer than 255 bytes"));
    }
    Ok(elem(hash_to_curve(t, msg, dst)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Elem>()?;
    m.add_class::<PreparedG2>()?;
    for f in [
        wrap_pyfunction!(bls_random, m)?, wrap_pyfunction!(bls_init, m)?, wrap_pyfunction!(bls_generator, m)?,
        wrap_pyfunction!(bls_order, m)?, wrap_pyfunction!(bls_deserialize, m)?, wrap_pyfunction!(bls_pair, m)?,
        wrap_pyfunction!(bls_multi_pair, m)?, wrap_pyfunction!(bls_gt_dlog, m)?, wrap_pyfunction!(bls_hash, m)?,
        wrap_pyfunction!(bls_deserialize_many, m)?, wrap_pyfunction!(bls_serialize_many, m)?,
        wrap_pyfunction!(bls_msm, m)?, wrap_pyfunction!(bls_msm_many, m)?, wrap_pyfunction!(bls_pow_many, m)?,
        wrap_pyfunction!(bls_gt_dlog_cache, m)?, wrap_pyfunction!(bls_prepare_g2, m)?,
        wrap_pyfunction!(bls_multi_pair_prepared, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
