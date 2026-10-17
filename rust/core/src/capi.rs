//! A small C ABI over the kernels, for the bit-identity tests of the Python
//! package (tests/test_rust_core.py loads it with ctypes) before the PyO3
//! bindings exist. Values mod q cross it as 16-byte little-endian integers.
//! Not a public interface: every pointer must be valid for the lengths given.

#![allow(clippy::missing_safety_doc)]

use crate::ntt::AnyRing;
use crate::rlwe::{self, EncKey, Randomness};
use crate::{with_ring, Word};
use std::slice;

unsafe fn u128s(p: *const u8, count: usize) -> Vec<u128> {
    slice::from_raw_parts(p, 16 * count)
        .chunks_exact(16)
        .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

unsafe fn put_u128s(p: *mut u8, vals: impl Iterator<Item = u128>) {
    for (i, v) in vals.enumerate() {
        std::ptr::copy_nonoverlapping(v.to_le_bytes().as_ptr(), p.add(16 * i), 16);
    }
}

unsafe fn materials<'a>(ptrs: *const *const u8, lens: *const usize, count: usize) -> Vec<&'a [u8]> {
    (0..count).map(|i| slice::from_raw_parts(*ptrs.add(i), *lens.add(i))).collect()
}

fn words<W: Word>(m: &crate::Modulus<W>, vals: &[u128]) -> Vec<W> {
    vals.iter().map(|&v| m.reduce_u128(v)).collect()
}

/// A ring for (q, n, root); null if unsupported.
#[no_mangle]
pub unsafe extern "C" fn pf4_ring_new(q: *const u8, n: usize, root: *const u8) -> *mut AnyRing {
    let (q, root) = (u128s(q, 1)[0], u128s(root, 1)[0]);
    match AnyRing::new(q, n, root) {
        Ok(r) => Box::into_raw(Box::new(r)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn pf4_ring_free(r: *mut AnyRing) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

#[no_mangle]
pub unsafe extern "C" fn pf4_word_bits(r: *const AnyRing) -> u32 {
    (*r).word_bits()
}

/// out = NTT(inp) (inverse != 0: inverse NTT); n values each.
#[no_mangle]
pub unsafe extern "C" fn pf4_ntt(r: *const AnyRing, inp: *const u8, out: *mut u8, inverse: i32) {
    with_ring!(&*r, ring => {
        let x = words(&ring.m, &u128s(inp, ring.n));
        let mut y = x.clone();
        if inverse != 0 { ring.inverse(&x, &mut y) } else { ring.forward(&x, &mut y) }
        put_u128s(out, y.iter().map(|v| v.to_u128()));
    })
}

#[no_mangle]
pub unsafe extern "C" fn pf4_gauss(buf: *const u8, len: usize, count: usize, sigma: f64, out: *mut i64) {
    crate::gauss::gauss_fill(slice::from_raw_parts(buf, len), sigma, slice::from_raw_parts_mut(out, count));
}

#[no_mangle]
pub unsafe extern "C" fn pf4_shake256(msg: *const u8, len: usize, out: *mut u8, out_len: usize) {
    crate::keccak::shake256(slice::from_raw_parts(msg, len), slice::from_raw_parts_mut(out, out_len));
}

#[no_mangle]
pub unsafe extern "C" fn pf4_hash_to_zq(r: *const AnyRing, mat: *const u8, len: usize, count: usize, out: *mut u8) {
    with_ring!(&*r, ring => {
        let mut w = vec![Word::ZERO; count];
        crate::keccak::hash_to_zq(&ring.m, slice::from_raw_parts(mat, len), &mut w);
        put_u128s(out, w.iter().map(|v| v.to_u128()));
    })
}

/// MIFERingLWE._encrypt_rows: ct0 (eta n) and ct1 (n) from w (eta n) and the random bytes.
#[no_mangle]
pub unsafe extern "C" fn pf4_encrypt_rows(
    r: *const AnyRing, a: *const u8, pk: *const u8, eta: usize, delta: *const u8, w: *const u8, sigma2: f64,
    sigma3: f64, rnd: *const u8, rnd_len: usize, ct0: *mut u8, ct1: *mut u8,
) {
    with_ring!(&*r, ring => {
        let n = ring.n;
        let pkv = words(&ring.m, &u128s(pk, eta * n));
        let rows: Vec<Vec<_>> = pkv.chunks(n).map(|c| c.to_vec()).collect();
        let key = EncKey::new(ring, &words(&ring.m, &u128s(a, n)), &rows, None, ring.m.reduce_u128(u128s(delta, 1)[0]));
        let wv = words(&ring.m, &u128s(w, eta * n));
        let (mut c0, mut c1) = (vec![Word::ZERO; eta * n], vec![Word::ZERO; n]);
        rlwe::encrypt_chunk(ring, &key, &wv, sigma2, sigma3, slice::from_raw_parts(rnd, rnd_len), &mut c0, &mut c1);
        put_u128s(ct0, c0.iter().map(|v| v.to_u128()));
        put_u128s(ct1, c1.iter().map(|v| v.to_u128()));
    })
}

/// MCFE ndarray encryption of one layer (eta = 1); out holds 2 n values per chunk.
/// Returns the number of chunks, or -1.
#[no_mangle]
pub unsafe extern "C" fn pf4_encrypt_tensor(
    r: *const AnyRing, a: *const u8, pk: *const u8, u: *const u8, delta: *const u8, x: *const i64, len: usize,
    mats: *const *const u8, mat_lens: *const usize, n_mats: usize, sigma2: f64, sigma3: f64, rnd: *const u8,
    rnd_len: usize, threads: usize, out: *mut u8,
) -> i64 {
    with_ring!(&*r, ring => {
        let n = ring.n;
        let key = EncKey::new(ring, &words(&ring.m, &u128s(a, n)), &[words(&ring.m, &u128s(pk, n))],
                              Some(&[words(&ring.m, &u128s(u, n))]), ring.m.reduce_u128(u128s(delta, 1)[0]));
        let mats = materials(mats, mat_lens, n_mats);
        let rand = if rnd.is_null() { Randomness::Os } else { Randomness::Given(slice::from_raw_parts(rnd, rnd_len)) };
        let outp = out as usize;
        let res = rlwe::encrypt_tensor(ring, &key, slice::from_raw_parts(x, len), Some(&mats), sigma2, sigma3, rand,
            threads, |j, c0, c1| {
                let p = (outp + 32 * n * j) as *mut u8;
                put_u128s(p, c0.iter().chain(c1.iter()).map(|v| v.to_u128()));
            });
        match res { Ok(c) => c as i64, Err(_) => -1 }
    })
}

/// One chunk of MIFE / MCFE decryption: S slots of (eta + 1) n values, weights
/// (S eta, plain), sk_y (S n, plain coefficients), z (n); k values decoded.
/// Returns 1, or 0 if a value does not fit an i64.
#[no_mangle]
pub unsafe extern "C" fn pf4_decrypt_chunk(
    r: *const AnyRing, eta: usize, s: usize, z: *const u8, slots: *const u8, ys: *const u8, sk: *const u8,
    delta: *const u8, bound: i64, k: usize, out: *mut i64, none: *mut u8,
) -> i32 {
    with_ring!(&*r, ring => {
        let n = ring.n;
        let m = &ring.m;
        let slotv = words(m, &u128s(slots, s * (eta + 1) * n));
        let ym: Vec<_> = words(m, &u128s(ys, s * eta)).iter().map(|&v| m.to_mont(v)).collect();
        let skv = words(m, &u128s(sk, s * n));
        let mut hats = vec![Word::ZERO; s * n];
        for (src, dst) in skv.chunks(n).zip(hats.chunks_mut(n)) {
            ring.forward_mont(src, dst);
        }
        let slot_refs: Vec<&[_]> = slotv.chunks((eta + 1) * n).collect();
        let hat_refs: Vec<&[_]> = hats.chunks(n).collect();
        let ok = rlwe::decrypt_chunk(ring, eta, &words(m, &u128s(z, n)), &slot_refs, &ym, &hat_refs,
            m.reduce_u128(u128s(delta, 1)[0]), bound as i128, slice::from_raw_parts_mut(out, k),
            slice::from_raw_parts_mut(none, k));
        ok as i32
    })
}

/// MCFE key terms: n_labels outputs of n values; materials label-major (n_labels S).
#[no_mangle]
pub unsafe extern "C" fn pf4_pad_terms(
    r: *const AnyRing, base: *const u8, mats: *const *const u8, mat_lens: *const usize, n_labels: usize, s: usize,
    ys: *const u8, eta: usize, threads: usize, out: *mut u8,
) {
    with_ring!(&*r, ring => {
        let n = ring.n;
        let m = &ring.m;
        let all = materials(mats, mat_lens, n_labels * s);
        let per_label: Vec<Vec<&[u8]>> = all.chunks(s).map(|c| c.to_vec()).collect();
        let ym: Vec<_> = words(m, &u128s(ys, s * eta)).iter().map(|&v| m.to_mont(v)).collect();
        let outp = out as usize;
        rlwe::pad_terms(ring, &words(m, &u128s(base, n)), &per_label, &ym, eta, threads, |l, zv| {
            put_u128s((outp + 16 * n * l) as *mut u8, zv.iter().map(|v| v.to_u128()));
        });
    })
}

/// Dense-LWE sums (see dense::dense_dots); out holds n_mats * sum_limbs(nb, kl) limbs.
#[no_mangle]
pub unsafe extern "C" fn pf4_dense_dots(
    mats: *const *const u8, mat_lens: *const usize, n_mats: usize, nb: usize, count: usize, key: *const u64,
    kl: usize, threads: usize, out: *mut u64,
) {
    let mats = materials(mats, mat_lens, n_mats);
    let res = crate::dense::dense_dots(&mats, nb, count, slice::from_raw_parts(key, count * kl), kl, threads);
    std::ptr::copy_nonoverlapping(res.as_ptr(), out, res.len());
}
