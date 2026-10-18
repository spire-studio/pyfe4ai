//! The replicated key sharing of the LWE threshold layers (pyfe4ai-core
//! `replicated`): dealing and the sums of pieces over Z_q for any odd q below
//! 2^253, without an NTT ring (the moduli of tMIFE-LWE and tMCFE-LWE have
//! none). Values cross as decimal strings or ints; anything else is a
//! TypeError, for which threshold_utils falls back to its Python code (which
//! then decides what such values mean). Secret words, digits and seed
//! materials are wiped before they are freed; the Python strings returned
//! cannot be.

use super::{index_mod, secret_byte_rows, u256_arg, wipe_byte_rows, wipe_rng, wipe_word_rows, CallDrbg, DecBuf};
use pyfe4ai_core::replicated;
use pyfe4ai_core::rlwe::wipe_words;
use pyfe4ai_core::{Modulus, Word, U256};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rand_chacha::rand_core::RngCore;
use zeroize::Zeroize;

/// The body with `$m` the Modulus of q in the word size it needs (u64 below
/// 2^63, u128 below 2^125, 256-bit words below 2^253).
macro_rules! with_zq {
    ($q:expr, $m:ident => $body:expr) => {{
        let q: U256 = $q;
        let bits = q.bits();
        if bits <= u64::MAX_Q_BITS {
            let $m = Modulus::<u64>::new(u64::from_u256(q));
            $body
        } else if bits <= u128::MAX_Q_BITS {
            let $m = Modulus::<u128>::new(u128::from_u256(q));
            $body
        } else {
            let $m = Modulus::<U256>::new(q);
            $body
        }
    }};
}

fn zq_arg(q: &Bound<'_, PyAny>) -> PyResult<U256> {
    let v = u256_arg(q)?;
    if v.bits() > U256::MAX_Q_BITS || v < U256::from_u128(3) || !v.bit0() {
        return Err(PyValueError::new_err("q must be odd, at least 3 and below 2^253"));
    }
    Ok(v)
}

/// Values mod q of plain decimal strings and ints (operator.index); a
/// TypeError for anything else.
fn read_exact<W: Word>(seq: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, m: &Modulus<W>) -> PyResult<Vec<W>> {
    let one = |ob: Bound<'_, PyAny>| -> PyResult<W> {
        if let Ok(s) = ob.cast::<PyString>() {
            return s
                .to_str()
                .ok()
                .and_then(|t| W::parse_dec(t.as_bytes(), m.q))
                .ok_or_else(|| PyTypeError::new_err("not a plain decimal string"));
        }
        index_mod(&ob, q, m).map_err(|_| PyTypeError::new_err("not an int or a plain decimal string"))
    };
    let mut out = Vec::new();
    let res = (|| {
        if let Ok(list) = seq.cast::<PyList>() {
            out.reserve(list.len());
            for ob in list.iter() {
                out.push(one(ob)?);
            }
            return Ok(());
        }
        for ob in seq.try_iter()? {
            out.push(one(ob?)?);
        }
        Ok(())
    })();
    match res {
        Ok(()) => Ok(out),
        Err(e) => {
            wipe_words(&mut out);
            Err(e)
        }
    }
}

/// Decimal strings of secret words; the words and the digits are wiped.
fn secret_list<'py, W: Word>(py: Python<'py>, words: &mut [W]) -> PyResult<Bound<'py, PyList>> {
    let mut dec = DecBuf::new(words);
    wipe_words(words);
    let out = dec.list(py);
    dec.text.zeroize();
    out
}

/// threshold_utils.share_replicated over flat values: `[correction, r_1, ...,
/// r_k]` as lists of decimal strings, the k = n_random random pieces drawn
/// piece by piece, value by value, as SystemRandom.randrange(0, q) draws them,
/// from the given bytes (tests; ValueError if they run out) or the call's
/// DRBG (stream 0, read sequentially), and the correction the values minus
/// their sum.
#[pyfunction]
#[pyo3(signature = (q, values, n_random, rnd, seed=None))]
fn zq_replicated_deal(py: Python<'_>, q: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>, n_random: usize,
                      rnd: &Bound<'_, PyAny>, seed: Option<Vec<u8>>) -> PyResult<Py<PyList>> {
    let qv = zq_arg(q)?;
    let mut given: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    // the values, the pieces, their digits and the caller's bytes are wiped after use, on every path
    let out = (|| -> PyResult<Py<PyList>> {
        let drbg = CallDrbg::new(seed.as_deref())?;
        with_zq!(qv, m => {
            let mut vals = read_exact(values, q, &m)?;
            let pieces = py.detach(|| match &given {
                Some(b) => {
                    let mut pos = 0;
                    replicated::deal(&m, &vals, n_random, |out: &mut [u8]| {
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
                    let res = replicated::deal(&m, &vals, n_random, |out: &mut [u8]| {
                        rng.fill_bytes(out);
                        true
                    });
                    wipe_rng(&mut rng);
                    res
                }
            });
            wipe_words(&mut vals);
            let mut pieces = pieces.ok_or_else(|| PyValueError::new_err("not enough random bytes"))?;
            let mut lists = Vec::with_capacity(pieces.len());
            let mut res = Ok(());
            for piece in pieces.iter_mut() {
                if res.is_ok() {
                    match secret_list(py, piece) {
                        Ok(l) => lists.push(l),
                        Err(e) => res = Err(e),
                    }
                }
                wipe_words(piece);
            }
            res?;
            Ok(PyList::new(py, lists)?.unbind())
        })
    })();
    if let Some(b) = given.as_mut() {
        pyfe4ai_core::rlwe::wipe_bytes(b);
    }
    out
}

/// The sum mod q of the rows (count values each: ints or plain decimal
/// strings) plus -- with `subtract`, minus -- hash_to_zq_vector(material, q,
/// count) (shake256-wide) of every material, as decimal strings: the sums of
/// threshold_utils.share_for_enrolled and _combine_pieces. TypeError for other
/// values, ValueError for rows of another length.
#[pyfunction]
#[pyo3(signature = (q, count, rows, materials, subtract=false))]
fn zq_sum(py: Python<'_>, q: &Bound<'_, PyAny>, count: usize, rows: &Bound<'_, PyAny>, materials: &Bound<'_, PyAny>,
          subtract: bool) -> PyResult<Py<PyList>> {
    let qv = zq_arg(q)?;
    let mut materials = secret_byte_rows(materials)?; // they hold the seeds: wiped on every path
    let out = with_zq!(qv, m => {
        let mut words: Vec<Vec<_>> = Vec::new();
        let read = (|| {
            for row in rows.try_iter()? {
                let row = read_exact(&row?, q, &m)?;
                let len = row.len();
                words.push(row);
                if len != count {
                    return Err(PyValueError::new_err(format!("expected rows of {} values", count)));
                }
            }
            Ok(())
        })();
        let res = read.and_then(|()| {
            let mats: Vec<&[u8]> = materials.iter().map(|v| v.as_slice()).collect();
            let mut acc = py.detach(|| replicated::sum(&m, count, &words, &mats, subtract));
            let list = secret_list(py, &mut acc).map(|l| l.unbind());
            wipe_words(&mut acc);
            list
        });
        wipe_word_rows(&mut words);
        res
    });
    wipe_byte_rows(&mut materials);
    out
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zq_replicated_deal, m)?)?;
    m.add_function(wrap_pyfunction!(zq_sum, m)?)?;
    Ok(())
}
