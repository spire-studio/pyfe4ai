//! Ring-LWE SIFE / MIFE (BMMS21): a slot's encryption key (EncKeyObj), the
//! encryption of rows (MIFERingLWE._encrypt_rows; SIFE is the one-slot case),
//! and the decryption of every chunk of a layer (decrypt_tensor), which the
//! MCFE schemes use too.

use crate::drbg::CallDrbg;
use crate::hashing::hash_mode;
use crate::ring::{get_ring, RingCtx};
use crate::rns;
use crate::sampling::{exact_err, exact_fn, with_noise, FacctArg};
use crate::secret::wipe_word_rows;
use crate::values::{
    bytes_to_words, dec_list, delta_mod, gather, py_index, read_rows, read_values, str_int_mod, GText, Gathered, SLOW,
};
use pyfe4ai_core::codec::{bin_read, bin_write};
use pyfe4ai_core::facct;
use pyfe4ai_core::gauss::gauss_bytes;
use pyfe4ai_core::rlwe::{self, EncKey, FacctNoise, Noise};
use pyfe4ai_core::{parallel_for, with_ring, AnyRing, Modulus, Ring, Word, U256};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::sync::{Arc, Mutex};

pub(crate) enum AnyKey {
    W64(EncKey<u64>),
    W128(EncKey<u128>),
    W256(EncKey<U256>),
}

/// A slot's encryption key: NTT(a), NTT(pk_row) in Montgomery form, u, Delta.
#[pyclass(frozen, module = "pyfe4ai._rust_native")]
pub(crate) struct EncKeyObj {
    pub(crate) ring: Arc<AnyRing>,
    q: Py<PyAny>,
    pub(crate) key: AnyKey,
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
pub(crate) use with_key;

#[pyfunction]
#[pyo3(signature = (ctx, a, pk, u, delta, hash_mode=0))]
pub(crate) fn enc_key(py: Python<'_>, ctx: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, pk: &Bound<'_, PyAny>,
           u: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>, hash_mode: u8) -> PyResult<Py<PyAny>> {
    let mode = self::hash_mode(hash_mode)?;
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        return Ok(Py::new(py, rns::enc_key(py, c, a, pk, u, delta, mode)?)?.into_any());
    }
    let ctx = ctx.cast::<RingCtx>()?;
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
            let mut u_rows = if u.is_none() {
                None
            } else {
                let mut rows = read_rows(u, Some(n), q, m)?;
                if rows.len() != pk_rows.len() {
                    wipe_word_rows(&mut rows);
                    return Err(PyValueError::new_err("one pad row per pk row"));
                }
                Some(rows)
            };
            let d = delta_mod(delta, q, m);
            let key = d.map(|d| EncKey::new($r, &av, &pk_rows, u_rows.as_deref(), d));
            if let Some(rows) = u_rows.as_mut() {
                wipe_word_rows(rows); // the key holds its own copy (wiped on drop)
            }
            let mut key = key?;
            key.hash_mode = mode;
            AnyKey::$variant(key)
        }};
    }
    let key = match &*ring {
        AnyRing::W64(r) => build!(r, W64),
        AnyRing::W128(r) => build!(r, W128),
        AnyRing::W256(r) => build!(r, W256),
    };
    Ok(Py::new(py, EncKeyObj { ring, q: q.clone().unbind(), key })?.into_any())
}

/// MIFERingLWE._encrypt_rows: (ct0, ct1) as decimal strings, or format-1 bytes.
#[pyfunction]
#[pyo3(signature = (key, w_rows, sigma2, sigma3, rnd, binary=false, seed=None, facct=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encrypt(py: Python<'_>, key: &Bound<'_, PyAny>, w_rows: &Bound<'_, PyAny>, sigma2: f64, sigma3: f64,
           rnd: &Bound<'_, PyAny>, binary: bool, seed: Option<Vec<u8>>, facct: FacctArg) -> PyResult<Py<PyAny>> {
    if let Ok(k) = key.cast::<rns::RnsKeyObj>() {
        return rns::encrypt(py, k, w_rows, sigma2, sigma3, rnd, binary, seed, facct);
    }
    let obj = key.cast::<EncKeyObj>()?.get();
    let ring = obj.ring.clone();
    let n = ring.n();
    let qob = obj.q.bind(py);
    with_key!(obj, r, k => {
        // the rows are x + u: the pad u is secret, so every copy is wiped
        let mut rows = read_rows(w_rows, Some(n), qob, &r.m)?;
        let eta = k.eta;
        if rows.len() != eta {
            wipe_word_rows(&mut rows);
            return Err(PyValueError::new_err("one plaintext row per pk row"));
        }
        let mut w: Vec<_> = rows.concat();
        wipe_word_rows(&mut rows);
        let (mut c0, mut c1) = (vec![Word::ZERO; eta * n], vec![Word::ZERO; n]);
        let res = (|| -> PyResult<()> {
            if facct.is_some() {
                let mut given: Option<Vec<u8>> = if rnd.is_none() { None } else { Some(rnd.extract()?) };
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
                let failed = bits.failed;
                drop(bits);
                if let Some(b) = given.as_mut() {
                    rlwe::wipe_bytes(b); // the caller's bytes (a test hook): the samples' source
                }
                if failed {
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
                    rlwe::wipe_bytes(&mut rb);
                    return Err(PyValueError::new_err("not enough random bytes"));
                }
                py.detach(|| rlwe::encrypt_chunk(r, k, &w, sigma2, sigma3, &rb, &mut c0, &mut c1));
                rlwe::wipe_bytes(&mut rb);
            }
            Ok(())
        })();
        rlwe::wipe_words(&mut w);
        res?;
        if binary {
            let mut out = Vec::new();
            bin_write(ring.qbits(), r.logn, eta, &c0, &c1, &mut out);
            return Ok(PyBytes::new(py, &out).into_any().unbind());
        }
        let rows_out = PyList::new(py, c0.chunks(n).map(|row| dec_list(py, row)).collect::<PyResult<Vec<_>>>()?)?;
        Ok(PyTuple::new(py, [rows_out.into_any(), dec_list(py, &c1)?.into_any()])?.into_any().unbind())
    })
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
pub(crate) fn decrypt_tensor(py: Python<'_>, ctx: &Bound<'_, PyAny>, chunks: &Bound<'_, PyAny>, zs: &Bound<'_, PyAny>,
                  ks: Vec<i64>, sk_hats: Vec<Vec<u8>>, ys: &Bound<'_, PyAny>, delta: &Bound<'_, PyAny>,
                  bound: &Bound<'_, PyAny>, threads: usize) -> PyResult<Py<PyAny>> {
    if let Ok(c) = ctx.cast::<rns::RnsCtx>() {
        let (ring, q) = rns::ring_of(c);
        return decrypt_tensor_w(py, &*ring, q.bind(py), chunks, zs, ks, sk_hats, ys, delta, bound, threads);
    }
    let ctx = ctx.cast::<RingCtx>()?;
    let q = ctx.get().q.bind(py).clone();
    let ring = get_ring(ctx);
    with_ring!(&*ring, r => decrypt_tensor_w(py, r, &q, chunks, zs, ks, sk_hats, ys, delta, bound, threads))
}

/// What decrypt_tensor_w needs of a ring: the arithmetic of the values (words
/// mod q), the dimensions, the NTT-domain keys and the chunk kernel.
pub(crate) trait TensorRing<W: Word>: Sync {
    type Hat: Copy + Send + Sync;
    fn modulus(&self) -> &Modulus<W>;
    fn dims(&self) -> (usize, u32);
    fn read_hat(&self, b: &[u8]) -> PyResult<Vec<Self::Hat>>;
    #[allow(clippy::too_many_arguments)]
    fn decrypt_chunk(&self, eta: usize, z: &[W], slots: &[&[W]], ym: &[W], hats: &[&[Self::Hat]], delta: W,
                     bound: i128, out: &mut [i64], none: &mut [u8]) -> bool;
}

impl<W: Word> TensorRing<W> for Ring<W> {
    type Hat = W;
    fn modulus(&self) -> &Modulus<W> {
        &self.m
    }
    fn dims(&self) -> (usize, u32) {
        (self.n, self.logn)
    }
    fn read_hat(&self, b: &[u8]) -> PyResult<Vec<W>> {
        bytes_to_words(b, self.n)
    }
    fn decrypt_chunk(&self, eta: usize, z: &[W], slots: &[&[W]], ym: &[W], hats: &[&[W]], delta: W, bound: i128,
                     out: &mut [i64], none: &mut [u8]) -> bool {
        rlwe::decrypt_chunk(self, eta, z, slots, ym, hats, delta, bound, out, none)
    }
}

#[allow(clippy::too_many_arguments)]
fn decrypt_tensor_w<W: Word, R: TensorRing<W>>(py: Python<'_>, r: &R, q: &Bound<'_, PyAny>, chunks: &Bound<'_, PyAny>,
                             zs: &Bound<'_, PyAny>, ks: Vec<i64>, sk_hats: Vec<Vec<u8>>, ys: &Bound<'_, PyAny>,
                             delta: &Bound<'_, PyAny>, bound: &Bound<'_, PyAny>, threads: usize)
    -> PyResult<Py<PyAny>> {
    let m = r.modulus();
    let (n, logn) = r.dims();
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
    let dw = delta_mod(delta, q, m)?;
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
    let hats: Vec<Vec<R::Hat>> = sk_hats.iter().map(|h| r.read_hat(h)).collect::<PyResult<_>>()?;
    let zw: Vec<Vec<W>> = zvals.into_iter().map(|z| z.into_iter().map(|v| v.unwrap()).collect()).collect();
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
        let hat_refs: Vec<&[R::Hat]> = hats.iter().map(|v| v.as_slice()).collect();
        let mut o = outs[j].lock().unwrap();
        let (ref mut vals, ref mut nn, ref mut ok) = *o;
        *ok = r.decrypt_chunk(eta, &zw[j], &slot_refs, &ym, &hat_refs, dw, bound_v, vals, nn);
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
