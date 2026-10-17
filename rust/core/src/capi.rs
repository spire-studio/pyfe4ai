//! A small C ABI over the kernels, for the bit-identity tests of the Python
//! package (tests/test_rust_core.py loads it with ctypes) before the PyO3
//! bindings exist. Values mod q cross it as 32-byte little-endian integers.
//! Not a public interface: every pointer must be valid for the lengths given.

#![allow(clippy::missing_safety_doc)]

use crate::ntt::AnyRing;
use crate::rlwe::{self, EncKey, Randomness};
use crate::{with_ring, Word, U256};
use std::slice;

const VB: usize = 32;

unsafe fn vals_in(p: *const u8, count: usize) -> Vec<U256> {
    slice::from_raw_parts(p, VB * count)
        .chunks_exact(VB)
        .map(|c| U256::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

unsafe fn put_vals<'a, W: Word>(p: *mut u8, vals: impl Iterator<Item = &'a W>) {
    for (i, v) in vals.enumerate() {
        std::ptr::copy_nonoverlapping(v.to_u256().to_le_bytes().as_ptr(), p.add(VB * i), VB);
    }
}

unsafe fn materials<'a>(ptrs: *const *const u8, lens: *const usize, count: usize) -> Vec<&'a [u8]> {
    (0..count).map(|i| slice::from_raw_parts(*ptrs.add(i), *lens.add(i))).collect()
}

fn words<W: Word>(m: &crate::Modulus<W>, vals: &[U256]) -> Vec<W> {
    vals.iter().map(|&v| m.reduce_u256(v)).collect()
}

/// A ring for (q, n, root); null if unsupported.
#[no_mangle]
pub unsafe extern "C" fn pf4_ring_new(q: *const u8, n: usize, root: *const u8) -> *mut AnyRing {
    let (q, root) = (vals_in(q, 1)[0], vals_in(root, 1)[0]);
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
        let x = words(&ring.m, &vals_in(inp, ring.n));
        let mut y = x.clone();
        if inverse != 0 { ring.inverse(&x, &mut y) } else { ring.forward(&x, &mut y) }
        put_vals(out, y.iter());
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
        put_vals(out, w.iter());
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
        let pkv = words(&ring.m, &vals_in(pk, eta * n));
        let rows: Vec<Vec<_>> = pkv.chunks(n).map(|c| c.to_vec()).collect();
        let key = EncKey::new(ring, &words(&ring.m, &vals_in(a, n)), &rows, None, ring.m.reduce_u256(vals_in(delta, 1)[0]));
        let wv = words(&ring.m, &vals_in(w, eta * n));
        let (mut c0, mut c1) = (vec![Word::ZERO; eta * n], vec![Word::ZERO; n]);
        rlwe::encrypt_chunk(ring, &key, &wv, sigma2, sigma3, slice::from_raw_parts(rnd, rnd_len), &mut c0, &mut c1);
        put_vals(ct0, c0.iter());
        put_vals(ct1, c1.iter());
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
        let key = EncKey::new(ring, &words(&ring.m, &vals_in(a, n)), &[words(&ring.m, &vals_in(pk, n))],
                              Some(&[words(&ring.m, &vals_in(u, n))]), ring.m.reduce_u256(vals_in(delta, 1)[0]));
        let mats = materials(mats, mat_lens, n_mats);
        let rand = if rnd.is_null() { Randomness::Os } else { Randomness::Given(slice::from_raw_parts(rnd, rnd_len)) };
        let outp = out as usize;
        let res = rlwe::encrypt_tensor(ring, &key, slice::from_raw_parts(x, len), Some(&mats), sigma2, sigma3, rand,
            threads, |j, c0, c1| {
                let p = (outp + 2 * VB * n * j) as *mut u8;
                put_vals(p, c0.iter().chain(c1.iter()));
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
        let slotv = words(m, &vals_in(slots, s * (eta + 1) * n));
        let ym: Vec<_> = words(m, &vals_in(ys, s * eta)).iter().map(|&v| m.to_mont(v)).collect();
        let skv = words(m, &vals_in(sk, s * n));
        let mut hats = vec![Word::ZERO; s * n];
        for (src, dst) in skv.chunks(n).zip(hats.chunks_mut(n)) {
            ring.forward_mont(src, dst);
        }
        let slot_refs: Vec<&[_]> = slotv.chunks((eta + 1) * n).collect();
        let hat_refs: Vec<&[_]> = hats.chunks(n).collect();
        let ok = rlwe::decrypt_chunk(ring, eta, &words(m, &vals_in(z, n)), &slot_refs, &ym, &hat_refs,
            m.reduce_u256(vals_in(delta, 1)[0]), bound as i128, slice::from_raw_parts_mut(out, k),
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
        let ym: Vec<_> = words(m, &vals_in(ys, s * eta)).iter().map(|&v| m.to_mont(v)).collect();
        let outp = out as usize;
        rlwe::pad_terms(ring, &words(m, &vals_in(base, n)), &per_label, &ym, eta, threads, |l, zv| {
            put_vals((outp + VB * n * l) as *mut u8, zv.iter());
        });
    })
}

/// One chunk of ThresholdMCFERingLWE.share_decrypt with its noise: S slots of
/// (eta + 1) n values, weights (S eta, plain), key shares (S n, plain
/// coefficients), z, Delta, lambda and the noise (n values mod q); w and c_y (n each).
#[no_mangle]
pub unsafe extern "C" fn pf4_share_partial(
    r: *const AnyRing, eta: usize, s: usize, slots: *const u8, ys: *const u8, sk: *const u8, z: *const u8,
    delta: *const u8, lambda: *const u8, noise: *const u8, w: *mut u8, cy: *mut u8,
) {
    with_ring!(&*r, ring => {
        let n = ring.n;
        let m = &ring.m;
        let slotv = words(m, &vals_in(slots, s * (eta + 1) * n));
        let ym: Vec<_> = words(m, &vals_in(ys, s * eta)).iter().map(|&v| m.to_mont(v)).collect();
        let skv = words(m, &vals_in(sk, s * n));
        let mut hats = vec![Word::ZERO; s * n];
        for (src, dst) in skv.chunks(n).zip(hats.chunks_mut(n)) {
            ring.forward_mont(src, dst);
        }
        let slot_refs: Vec<&[_]> = slotv.chunks((eta + 1) * n).collect();
        let hat_refs: Vec<&[_]> = hats.chunks(n).collect();
        let (mut wv, mut cyv) = (vec![Word::ZERO; n], vec![Word::ZERO; n]);
        rlwe::share_partial(ring, eta, &slot_refs, &ym, &hat_refs, &words(m, &vals_in(z, n)),
            m.reduce_u256(vals_in(delta, 1)[0]), m.reduce_u256(vals_in(lambda, 1)[0]), &words(m, &vals_in(noise, n)),
            &mut wv, &mut cyv);
        put_vals(w, wv.iter());
        put_vals(cy, cyv.iter());
    })
}

/// Shamir shares (threshold t) of `count` secrets at `n_points` points, the
/// coefficients drawn from `rnd` as SystemRandom.randrange draws them; out holds
/// n_points * count values (point-major). Returns 1, or 0 if the bytes run out.
#[no_mangle]
pub unsafe extern "C" fn pf4_shamir_share(
    r: *const AnyRing, secrets: *const u8, count: usize, t: usize, points: *const u64, n_points: usize,
    rnd: *const u8, rnd_len: usize, out: *mut u8,
) -> i32 {
    with_ring!(&*r, ring => {
        let m = &ring.m;
        let bytes = slice::from_raw_parts(rnd, rnd_len);
        let mut pos = 0;
        let res = crate::shamir::share(m, &words(m, &vals_in(secrets, count)), t, slice::from_raw_parts(points, n_points),
            |b: &mut [u8]| {
                if pos + b.len() > bytes.len() {
                    return false;
                }
                b.copy_from_slice(&bytes[pos..pos + b.len()]);
                pos += b.len();
                true
            });
        match res {
            Some(shares) => {
                for (p, sh) in shares.iter().enumerate() {
                    put_vals(out.add(VB * count * p), sh.iter());
                }
                1
            }
            None => 0,
        }
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

/// RFC 9380 expand_message_xmd with SHA-512 to 64 bytes.
#[no_mangle]
pub unsafe extern "C" fn pf4_xmd64(msg: *const u8, len: usize, dst: *const u8, dst_len: usize, out: *mut u8) {
    let o = crate::sha512::expand_message_xmd_64(slice::from_raw_parts(msg, len), slice::from_raw_parts(dst, dst_len));
    std::ptr::copy_nonoverlapping(o.as_ptr(), out, 64);
}
