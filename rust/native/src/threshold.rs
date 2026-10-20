//! The threshold Ring-LWE MCFE (pyfe4ai.schemes.mcfe.ring_lwe_threshold):
//! Shamir dealing of key terms (shamir_share), the JSON path (share_inputs,
//! share_partial, share_combine over decimal-string ciphertexts) and the
//! binary paths (pyfe4ai-core `threshold`): partial decryptions of format-2
//! ciphertexts (`ct_format = "binary"` public parameters) with outputs as
//! bytes, their combination, and the uniform flooding noise, which the JSON
//! path's share_partial draws too.
//! Values of key shares may be decimal strings, ints or fixed-width bytes
//! (read_poly); the noise, the key part and the caller's noise material
//! are wiped before they are freed.

use crate::drbg::{wipe_rng, CallDrbg};
use crate::ring::{get_ring, RingCtx};
use crate::secret::wipe_word_rows;
use crate::values::{
    bytes_to_words, delta_mod, gather, index_mod, int_mod_fast, py_index, read_rows, read_values, word_pyint,
    words_to_bytes, DecBuf, SLOW,
};
use pyfe4ai_core::rlwe::{self, wipe_bytes, wipe_words};
pub(crate) use pyfe4ai_core::threshold::value_bytes;
use pyfe4ai_core::threshold;
use pyfe4ai_core::{with_ring, Modulus, Word};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use rand_chacha::rand_core::RngCore;

/// Words that hold secrets (key shares, NTT(key share), the noise), zeroed
/// when dropped: on every path out of a binding, errors included.
pub(crate) struct SecretWords<W: Word>(pub Vec<W>);

impl<W: Word> Drop for SecretWords<W> {
    fn drop(&mut self) {
        wipe_words(&mut self.0);
    }
}

impl<W: Word> std::ops::Deref for SecretWords<W> {
    type Target = [W];
    fn deref(&self) -> &[W] {
        &self.0
    }
}

/// Bytes that hold secrets (copies of NTT(key share), the noise key's
/// material), zeroed when dropped.
pub(crate) struct SecretBytes(pub Vec<u8>);

impl Drop for SecretBytes {
    fn drop(&mut self) {
        wipe_bytes(&mut self.0);
    }
}

/// The bytes objects of `rows` copied into SecretBytes (wiped however the
/// call ends).
pub(crate) fn secret_rows(rows: &Bound<'_, PyAny>) -> PyResult<Vec<SecretBytes>> {
    let mut out = Vec::new();
    for r in rows.try_iter()? {
        out.push(SecretBytes(r?.extract::<Vec<u8>>()?));
    }
    Ok(out)
}

/// NTT(key share) words of `rows` (n words each), wiped however the call ends.
pub(crate) fn secret_hats<W: Word>(rows: &[SecretBytes], n: usize) -> PyResult<Vec<SecretWords<W>>> {
    rows.iter().map(|b| bytes_to_words(&b.0, n).map(SecretWords)).collect()
}

/// n values mod q: a sequence as read_values reads it, or bytes of n
/// ceil(bits(q) / 8)-byte little-endian words, each below q.
pub(crate) fn read_poly<W: Word>(ob: &Bound<'_, PyAny>, n: usize, q: &Bound<'_, PyAny>, m: &Modulus<W>)
    -> PyResult<Vec<W>> {
    if let Ok(b) = ob.cast::<PyBytes>() {
        let mut out = vec![W::ZERO; n];
        if threshold::read_strict(b.as_bytes(), m.q, value_bytes(m.bits), &mut out).is_none() {
            wipe_words(&mut out);
            return Err(PyValueError::new_err(format!(
                "expected {} values below q as {}-byte words", n, value_bytes(m.bits))));
        }
        return Ok(out);
    }
    read_values(ob, Some(n), q, m, false)
}

/// The flooding noise of a partial decryption: `count` values of the given
/// list, or the uniform noise of `flood = (bits, material)` drawn here (the
/// caller holds the material in a SecretBytes).
pub(crate) fn noise_values<W: Word>(noise: Option<&Bound<'_, PyAny>>, flood: Option<(u32, &[u8])>, count: usize,
                                    q: &Bound<'_, PyAny>, m: &Modulus<W>) -> PyResult<SecretWords<W>> {
    match (noise, flood) {
        (None, Some((bits, material))) => {
            let mut out = SecretWords(vec![W::ZERO; count]);
            threshold::uniform_flooding(m, bits, material, &mut out.0).map_err(PyValueError::new_err)?;
            Ok(out)
        }
        (Some(list), None) => read_values(list, Some(count), q, m, false).map(SecretWords),
        _ => Err(PyValueError::new_err("give the noise or the uniform flooding (bits, material), not both")),
    }
}

/// The `flood` argument with its material held in a SecretBytes.
pub(crate) fn secret_flood(flood: Option<(u32, Vec<u8>)>) -> Option<(u32, SecretBytes)> {
    flood.map(|(bits, material)| (bits, SecretBytes(material)))
}

/// ceil(bits(q) / 8), the bytes of a value in the binary formats.
pub(crate) fn value_bytes_of<W: Word>(m: &Modulus<W>) -> usize {
    value_bytes(m.bits)
}

/// Secret values (key shares) as wb-byte little-endian words (the caller
/// wipes its words; the returned bytes object cannot be).
pub(crate) fn secret_bytes<W: Word>(py: Python<'_>, vals: &[W], wb: usize) -> Py<PyBytes> {
    values_bytes(py, vals, wb)
}

pub(crate) fn values_bytes_pub<W: Word>(py: Python<'_>, vals: &[W], wb: usize) -> Py<PyBytes> {
    values_bytes(py, vals, wb)
}

fn values_bytes<W: Word>(py: Python<'_>, vals: &[W], wb: usize) -> Py<PyBytes> {
    PyBytes::new_with(py, vals.len() * wb, |out| {
        for (o, v) in out.chunks_exact_mut(wb).zip(vals) {
            v.write_le(o);
        }
        Ok(())
    })
    .expect("allocating bytes")
    .unbind()
}

// ------------------------------------------------- dealing and the JSON path

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
pub(crate) fn share_inputs(py: Python<'_>, ctx: &Bound<'_, RingCtx>, keys: Vec<String>, ks: Vec<i64>, cts: &Bound<'_, PyAny>,
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
/// the Lagrange coefficient and the noise (n values), or `flood = (bits,
/// material)` for the uniform noise drawn here (threshold::uniform_flooding).
#[pyfunction]
#[pyo3(signature = (ctx, packed, eta, ys, sk_hats, z, delta, lagrange, noise=None, flood=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn share_partial(py: Python<'_>, ctx: &Bound<'_, RingCtx>, packed: Vec<Vec<u8>>, eta: usize, ys: &Bound<'_, PyAny>,
                 sk_hats: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>, lagrange: &Bound<'_, PyAny>,
                 noise: Option<&Bound<'_, PyAny>>, flood: Option<(u32, Vec<u8>)>) -> PyResult<(Py<PyList>, Py<PyList>)> {
    // the secrets (the noise key's material, NTT(key share) copies, z, the noise) are wiped on every path
    let flood = secret_flood(flood);
    let sk_hats = secret_rows(sk_hats)?;
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    if packed.is_empty() || packed.len() != sk_hats.len() || eta == 0 {
        return Err(PyValueError::new_err("one ciphertext and key share per client"));
    }
    with_ring!(&*ring, r => {
        let (n, m) = (r.n, &r.m);
        let slots: Vec<Vec<_>> = packed.iter().map(|b| bytes_to_words(b, (eta + 1) * n)).collect::<PyResult<_>>()?;
        let hats = secret_hats(&sk_hats, n)?;
        let y_rows = read_rows(ys, Some(eta), &q, m)?;
        if y_rows.len() != slots.len() {
            return Err(PyValueError::new_err("one weight row per client"));
        }
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
        let zv = SecretWords(read_poly(z, n, &q, m)?);
        let ev = noise_values(noise, flood.as_ref().map(|(b, mat)| (*b, mat.0.as_slice())), n, &q, m)?;
        let (d, l) = (delta_mod(delta, &q, m)?, index_mod(lagrange, &q, m)?);
        let (mut w, mut cy) = (vec![Word::ZERO; n], vec![Word::ZERO; n]);
        py.detach(|| {
            let sr: Vec<&[_]> = slots.iter().map(|v| v.as_slice()).collect();
            let hr: Vec<&[_]> = hats.iter().map(|v| &v[..]).collect();
            rlwe::share_partial(r, eta, &sr, &ym, &hr, &zv, d, l, &ev, &mut w, &mut cy);
        });
        Ok((DecBuf::new(&w).list(py)?.unbind(), DecBuf::new(&cy).list(py)?.unbind()))
    })
}

/// ThresholdMCFERingLWE._combine_group: decode_scaled of c_y - sum_k w_k mod q
/// for the first k values (ints, None where |mu| > bound); None when a value
/// needs Python (see read_plain) or a result does not fit an int64.
#[pyfunction]
pub(crate) fn share_combine(py: Python<'_>, ctx: &Bound<'_, RingCtx>, c_y: &Bound<'_, PyAny>, ws: &Bound<'_, PyAny>,
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
        let d = delta_mod(delta, &q, m)?;
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
/// With `binary`, one bytes object per point instead: the shares as
/// ceil(bits(q) / 8)-byte little-endian words.
#[pyfunction]
#[pyo3(signature = (ctx, values, t, points, rnd, seed=None, binary=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shamir_share(py: Python<'_>, ctx: &Bound<'_, RingCtx>, values: &Bound<'_, PyAny>, t: usize, points: Vec<u64>,
                rnd: &Bound<'_, PyAny>, seed: Option<Vec<u8>>, binary: bool) -> PyResult<Py<PyList>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    if t < 1 {
        return Err(PyValueError::new_err("the threshold must be at least 1"));
    }
    let mut given: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
    let drbg = CallDrbg::new(seed.as_deref())?;
    // the secrets (key coefficients), the shares and the caller's bytes are wiped after use
    let out = (|| -> PyResult<Py<PyList>> {
        with_ring!(&*ring, r => {
            let m = &r.m;
            // binary key terms: one bytes object of ceil(bits(q) / 8)-byte words, each below q
            let mut secrets = match values.cast::<PyBytes>() {
                Ok(b) => {
                    let wb = value_bytes_of(m);
                    if b.as_bytes().len() % wb != 0 {
                        return Err(PyValueError::new_err("not a whole number of values"));
                    }
                    read_poly(values, b.as_bytes().len() / wb, &q, m)?
                }
                Err(_) => read_values(values, None, &q, m, false)?,
            };
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
            rlwe::wipe_words(&mut secrets);
            let mut shares = shares.ok_or_else(|| PyValueError::new_err("not enough random bytes"))?;
            if binary {
                let wb = value_bytes_of(m);
                let blobs: Vec<Py<PyBytes>> = shares.iter().map(|s| secret_bytes(py, s, wb)).collect();
                wipe_word_rows(&mut shares);
                return Ok(PyList::new(py, blobs)?.unbind());
            }
            let lists = shares.iter().map(|s| DecBuf::new(s).list(py)).collect::<PyResult<Vec<_>>>();
            wipe_word_rows(&mut shares);
            Ok(PyList::new(py, lists?)?.unbind())
        })
    })();
    if let Some(b) = given.as_mut() {
        rlwe::wipe_bytes(b);
    }
    out
}

// ------------------------------------------------------------ binary paths

/// ring_lwe_threshold.uniform_flooding_poly(bits, count, seed) mod q, for the
/// material domain || len(seed) || seed (tests compare it with the Python code).
#[pyfunction]
fn uniform_flooding(py: Python<'_>, ctx: &Bound<'_, RingCtx>, bits: u32, material: Vec<u8>, count: usize)
    -> PyResult<Py<PyList>> {
    let material = SecretBytes(material);
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => {
        let v = noise_values(None, Some((bits, &material.0)), count, &q, &r.m)?;
        let items = v.iter().map(|&x| word_pyint(py, x)).collect::<PyResult<Vec<_>>>();
        Ok(PyList::new(py, items?)?.unbind())
    })
}

/// ThresholdMCFERingLWE.share_decrypt of one chunk of format-2 ciphertexts
/// (`cts`, one per client in the order of `ys` and `sk_hats`): (w, c_y, k), w
/// and c_y the first k values as ceil(bits(q) / 8)-byte words. `z` is the
/// key share's term (n values: a sequence or bytes), the noise k values (or
/// `flood` for the uniform noise drawn here). ValueError("malformed
/// ciphertext", i) for a ciphertext that is not a strict format-2 one of
/// these parameters, and for different batch lengths.
#[pyfunction]
#[pyo3(signature = (ctx, cts, eta, ys, sk_hats, z, delta, lagrange, noise=None, flood=None))]
#[allow(clippy::too_many_arguments)]
fn share_partial_bin(py: Python<'_>, ctx: &Bound<'_, RingCtx>, cts: Vec<Vec<u8>>, eta: usize, ys: &Bound<'_, PyAny>,
                     sk_hats: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>,
                     lagrange: &Bound<'_, PyAny>, noise: Option<&Bound<'_, PyAny>>, flood: Option<(u32, Vec<u8>)>)
    -> PyResult<(Py<PyBytes>, Py<PyBytes>, usize)> {
    // the secrets (the noise key's material, NTT(key share) copies, z, the noise) are wiped on every path
    let flood = secret_flood(flood);
    let sk_hats = secret_rows(sk_hats)?;
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    if cts.is_empty() || cts.len() != sk_hats.len() || eta == 0 {
        return Err(PyValueError::new_err("one ciphertext and key share per client"));
    }
    with_ring!(&*ring, r => {
        let (n, m) = (r.n, &r.m);
        let mut parsed = Vec::with_capacity(cts.len());
        for (i, ct) in cts.iter().enumerate() {
            match threshold::read_ciphertext(ct, m.q, eta, r.logn) {
                Some(p) => parsed.push(p),
                None => return Err(PyValueError::new_err(("malformed ciphertext", i))),
            }
        }
        let k = parsed[0].0;
        if let Some(i) = parsed.iter().position(|p| p.0 != k) {
            return Err(PyValueError::new_err(("malformed ciphertext", i)));
        }
        let hats = secret_hats(&sk_hats, n)?;
        let y_rows = read_rows(ys, Some(eta), &q, m)?;
        if y_rows.len() != parsed.len() {
            return Err(PyValueError::new_err("one weight row per client"));
        }
        let ym: Vec<_> = y_rows.iter().flatten().map(|&v| m.to_mont(v)).collect();
        let zv = SecretWords(read_poly(z, n, &q, m)?);
        let ev = noise_values(noise, flood.as_ref().map(|(b, mat)| (*b, mat.0.as_slice())), k, &q, m)?;
        let (d, l) = (delta_mod(delta, &q, m)?, index_mod(lagrange, &q, m)?);
        let (mut w, mut cy) = (vec![Word::ZERO; k], vec![Word::ZERO; k]);
        py.detach(|| {
            let c0: Vec<&[_]> = parsed.iter().map(|p| p.1.as_slice()).collect();
            let c1: Vec<&[_]> = parsed.iter().map(|p| p.2.as_slice()).collect();
            let hr: Vec<&[_]> = hats.iter().map(|v| &v[..]).collect();
            threshold::share_partial_k(r, eta, &c0, &c1, &ym, &hr, &zv, d, l, &ev, &mut w, &mut cy);
        });
        let wb = value_bytes(m.bits);
        Ok((values_bytes(py, &w, wb), values_bytes(py, &cy, wb), k))
    })
}

/// ThresholdMCFERingLWE._combine_group of binary partial decryptions: the k
/// decoded values (ints, None where |mu| > bound), or None if one does not fit
/// an int64. ValueError unless c_y and every w are k strict values.
#[pyfunction]
fn share_combine_bin(py: Python<'_>, ctx: &Bound<'_, RingCtx>, c_y: &[u8], ws: Vec<Vec<u8>>, delta: &Bound<'_, PyAny>,
                     bound: &Bound<'_, PyAny>, k: usize) -> PyResult<Option<Py<PyList>>> {
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    let bound_v: i128 = match py_index(bound)?.extract::<i64>() {
        Ok(b) => b as i128,
        Err(_) => if py_index(bound)?.gt(0)? { 1i128 << 126 } else { -1 },
    };
    with_ring!(&*ring, r => {
        let m = &r.m;
        let wb = value_bytes(m.bits);
        if k < 1 || k > r.n {
            return Err(PyValueError::new_err("malformed partial decryption"));
        }
        let mut cy = vec![Word::ZERO; k];
        threshold::read_strict(c_y, m.q, wb, &mut cy).ok_or_else(|| PyValueError::new_err("malformed partial decryption"))?;
        let mut wv = Vec::with_capacity(ws.len());
        for w in ws.iter() {
            let mut v = vec![Word::ZERO; k];
            threshold::read_strict(w, m.q, wb, &mut v).ok_or_else(|| PyValueError::new_err("malformed partial decryption"))?;
            wv.push(v);
        }
        let d = delta_mod(delta, &q, m)?;
        let (mut out, mut none) = (vec![0i64; k], vec![0u8; k]);
        let wr: Vec<&[_]> = wv.iter().map(|v| v.as_slice()).collect();
        if !py.detach(|| threshold::combine(m, &cy, &wr, d, bound_v, &mut out, &mut none)) {
            return Ok(None);
        }
        let items = out.iter().zip(none.iter())
            .map(|(&v, &nn)| if nn == 1 { Ok(py.None().into_bound(py)) } else { Ok(v.into_pyobject(py)?.into_any()) })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some(PyList::new(py, items)?.unbind()))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniform_flooding, m)?)?;
    m.add_function(wrap_pyfunction!(share_partial_bin, m)?)?;
    m.add_function(wrap_pyfunction!(share_combine_bin, m)?)?;
    Ok(())
}
