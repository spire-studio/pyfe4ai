//! BLS12-381 for the pairing-based schemes (arkworks), with the element
//! interface the schemes use from charm-crypto: G1, G2 and GT written
//! multiplicatively (`*` the group operation, `**` exponentiation, `/`), ZR the
//! scalar field (`+ - * / **`, `int()`), `==` and `hash()` on all of them.
//! pyfe4ai.utils.bls12_381 wraps this into a PairingGroup.
//!
//! Elements are kept in memory as curve points (projective) / field elements;
//! serialization is the arkworks compressed form, deserialization checks
//! the encoding and subgroup membership. Random elements are g^r for scalars
//! from 64 OS-random bytes reduced mod r. Exponentiations and products of
//! powers (`bls_msm`, `bls_pow_many`) use the methods of `msm.rs`. Not
//! constant-time in general.

use crate::msm;
use ark_bls12_381::{Bls12_381, Fq12, Fr, G1Projective, G2Projective};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{CurveGroup, PrimeGroup};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use pyfe4ai_core::osrand;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyInt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// charm's type codes.
pub const ZR: u8 = 0;
pub const G1: u8 = 1;
pub const G2: u8 = 2;
pub const GT: u8 = 3;

type Gt = PairingOutput<Bls12_381>;

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

/// e(g1, g2) for the fixed generators.
fn gt_generator() -> Gt {
    static GT_GEN: OnceLock<Gt> = OnceLock::new();
    *GT_GEN.get_or_init(|| Bls12_381::pairing(G1Projective::generator(), G2Projective::generator()))
}

/// The power of a group element (not ZR) by e.
fn group_pow(v: &Val, e: &Fr) -> Val {
    match v {
        Val::G1(p) => Val::G1(msm::curve_msm(&[*p], &[*e])),
        Val::G2(p) => Val::G2(msm::curve_msm(&[*p], &[*e])),
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

fn deserialize_val(t: u8, data: &[u8]) -> PyResult<Val> {
    let bad = |_| PyValueError::new_err("invalid BLS12-381 encoding");
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
        let e = fr_from_int(exponent)?;
        Ok(elem(match self.v {
            Val::Zr(a) => {
                // ZR ** k with k an integer: k mod (r - 1) would be wrong for k < 0; use the
                // field power for k >= 0 and the inverse for k < 0
                let neg = exponent.lt(0).unwrap_or(false) && other_elem(exponent).is_none();
                let mag = if neg { -e } else { e };
                let p = a.pow(mag.into_bigint());
                if neg {
                    Val::Zr(p.inverse().ok_or_else(|| PyZeroDivisionError::new_err("0 ** negative in ZR"))?)
                } else {
                    Val::Zr(p)
                }
            }
            v => group_pow(&v, &e),
        }))
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
fn bls_deserialize_many(py: Python<'_>, items: Vec<(u8, Vec<u8>)>) -> PyResult<Vec<Elem>> {
    let vals: Vec<PyResult<Val>> = py.detach(|| {
        let mut seen: HashMap<(u8, &[u8]), Val> = HashMap::new();
        items
            .iter()
            .map(|(t, data)| {
                if let Some(v) = seen.get(&(*t, data.as_slice())) {
                    return Ok(*v);
                }
                let v = deserialize_val(*t, data)?;
                seen.insert((*t, data.as_slice()), v);
                Ok(v)
            })
            .collect()
    });
    vals.into_iter().map(|v| v.map(elem)).collect()
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

/// prod_i bases[i] ** scalars[i] for bases of one group (G1, G2 or GT) and
/// scalars ints or ZR elements: one multi-scalar multiplication.
#[pyfunction]
fn bls_msm(py: Python<'_>, bases: Vec<Bound<'_, Elem>>, scalars: &Bound<'_, PyAny>) -> PyResult<Elem> {
    let vals = group_bases(&bases)?;
    let ks = frs_from_seq(scalars)?;
    if ks.len() != vals.len() {
        return Err(PyValueError::new_err("msm takes as many scalars as bases"));
    }
    // group_bases checked that every base is of the group of the first
    Ok(elem(py.detach(|| match vals[0] {
        Val::G1(_) => {
            let ps: Vec<_> = vals.iter().filter_map(|v| if let Val::G1(p) = v { Some(*p) } else { None }).collect();
            Val::G1(msm::curve_msm(&ps, &ks))
        }
        Val::G2(_) => {
            let ps: Vec<_> = vals.iter().filter_map(|v| if let Val::G2(p) = v { Some(*p) } else { None }).collect();
            Val::G2(msm::curve_msm(&ps, &ks))
        }
        Val::Gt(_) => {
            let fs: Vec<_> = vals.iter().filter_map(|v| if let Val::Gt(t) = v { Some(t.0) } else { None }).collect();
            Val::Gt(PairingOutput(msm::gt_msm(&fs, &ks)))
        }
        Val::Zr(_) => unreachable!("group_bases rejects ZR"),
    })))
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

/// prod_i e(a_i, b_i): one Miller loop product and one final exponentiation.
/// Each pair is a G1 and a G2 element in either order, as for pair().
#[pyfunction]
fn bls_multi_pair(py: Python<'_>, a: Vec<Bound<'_, Elem>>, b: Vec<Bound<'_, Elem>>) -> PyResult<Elem> {
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
    Ok(elem(Val::Gt(py.detach(|| {
        Bls12_381::multi_pairing(G1Projective::normalize_batch(&ps), G2Projective::normalize_batch(&qs))
    }))))
}

// ----------------------------------------------------------- dlog in GT

struct GtTable {
    m: u64,
    giant: Gt,
    baby: HashMap<Vec<u8>, u64>,
}

const BABY_FACTOR: u64 = 16;

fn gt_table(base: Gt, bound: u64) -> Arc<GtTable> {
    type Tables = Mutex<HashMap<(Vec<u8>, u64), Arc<GtTable>>>;
    static TABLES: OnceLock<Tables> = OnceLock::new();
    let tables = TABLES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (serialize_val(&Val::Gt(base)), bound);
    if let Some(t) = tables.lock().unwrap().get(&key) {
        return t.clone();
    }
    let span = 2 * bound + 1;
    let m = span.min(((span as f64).sqrt() as u64 + 1) * BABY_FACTOR);
    let mut baby = HashMap::with_capacity(m as usize);
    let mut cur = Gt::zero();
    for j in 0..m {
        baby.insert(serialize_val(&Val::Gt(cur)), j);
        cur += base;
    }
    let t = Arc::new(GtTable { m, giant: -(base * Fr::from(m)), baby });
    tables.lock().unwrap().insert(key, t.clone());
    t
}

/// The x with base^x = target and |x| <= bound, or None (table cached per base and bound).
#[pyfunction]
fn bls_gt_dlog(py: Python<'_>, base: &Bound<'_, Elem>, target: &Bound<'_, Elem>, bound: u64) -> PyResult<Option<i64>> {
    let (g, h) = match (base.get().v, target.get().v) {
        (Val::Gt(g), Val::Gt(h)) => (g, h),
        _ => return Err(PyTypeError::new_err("dlog in GT takes GT elements")),
    };
    Ok(py.detach(|| {
        let t = gt_table(g, bound);
        let mut gamma = h + g * Fr::from(bound);
        for i in 0..(2 * bound) / t.m + 2 {
            if let Some(&j) = t.baby.get(&serialize_val(&Val::Gt(gamma))) {
                let x = (i * t.m + j) as i64 - bound as i64;
                return if x.unsigned_abs() <= bound { Some(x) } else { None };
            }
            gamma += t.giant;
        }
        None
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
    for f in [
        wrap_pyfunction!(bls_random, m)?, wrap_pyfunction!(bls_init, m)?, wrap_pyfunction!(bls_generator, m)?,
        wrap_pyfunction!(bls_order, m)?, wrap_pyfunction!(bls_deserialize, m)?, wrap_pyfunction!(bls_pair, m)?,
        wrap_pyfunction!(bls_multi_pair, m)?, wrap_pyfunction!(bls_gt_dlog, m)?, wrap_pyfunction!(bls_hash, m)?,
        wrap_pyfunction!(bls_deserialize_many, m)?, wrap_pyfunction!(bls_msm, m)?, wrap_pyfunction!(bls_pow_many, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
