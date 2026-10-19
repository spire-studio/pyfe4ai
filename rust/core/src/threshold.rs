//! Threshold Ring-LWE MCFE (pyfe4ai.schemes.mcfe.ring_lwe_threshold): the
//! uniform flooding noise, binary format 2 of its ciphertexts and partial
//! decryptions (ring_lwe_utils.encode_threshold_ciphertext), and a partial
//! decryption whose ct0 rows, outputs and noise cover the first k
//! coefficients only. Every function computes the values of the Python code.

use crate::arith::{Modulus, Word};
use crate::keccak::{shake256, wipe_long};
use crate::ntt::Ring;
use crate::rlwe::{decode_scaled_w, wipe_words};

/// Magic and version of binary format 2 (version 1 is MIFE / MCFE's format).
pub const BIN2_MAGIC: &[u8; 4] = b"PFRL";
pub const BIN2_VERSION: u8 = 2;
pub const BIN2_HEADER: usize = 14;

/// Bytes per value of the binary formats: ceil(bits(q) / 8).
pub fn value_bytes(qbits: u32) -> usize {
    ((qbits + 7) / 8) as usize
}

/// ring_lwe_threshold.uniform_flooding_poly mod q: value i is block i of
/// ceil((bits + 1) / 8) big-endian bytes of SHAKE-256(material), its low
/// bits + 1 bits, minus 2^bits. Needs 2^bits < q. The stream is wiped (the
/// material holds the server's noise key).
pub fn uniform_flooding<W: Word>(m: &Modulus<W>, bits: u32, material: &[u8], out: &mut [W]) -> Result<(), &'static str> {
    if bits == 0 || bits >= m.bits {
        return Err("the flooding width must satisfy 1 <= bits and 2^bits < q");
    }
    let nb = (bits as usize + 8) / 8;
    let top = 8 * nb - (bits as usize + 1); // high bits of the first byte to drop
    let mut stream = vec![0u8; nb * out.len()];
    shake256(material, &mut stream);
    let offset = W::ONE.shl(bits);
    let mut block = [0u8; 32];
    for (o, c) in out.iter_mut().zip(stream.chunks_exact(nb)) {
        block[..nb].copy_from_slice(c);
        block[0] &= 0xFFu8 >> top;
        let v = W::read_be(&block[..nb]);
        *o = if v >= offset { v.wsub(offset) } else { m.q.wsub(offset.wsub(v)) };
    }
    wipe_long(&mut stream);
    crate::rlwe::wipe_bytes(&mut block);
    Ok(())
}

/// The fields of a format-2 header: (wb, eta, logn, k).
fn header(data: &[u8]) -> Option<(usize, usize, u32, usize)> {
    if data.len() < BIN2_HEADER || &data[..4] != BIN2_MAGIC || data[4] != BIN2_VERSION || data[9] != 0 {
        return None;
    }
    let k = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
    Some((data[5] as usize, data[6] as usize | (data[7] as usize) << 8, data[8] as u32, k))
}

/// Values of wb-byte little-endian words, each < q (strict: a value >= q is
/// another encoding of a value below q, so it is refused).
pub fn read_strict<W: Word>(data: &[u8], q: W, wb: usize, out: &mut [W]) -> Option<()> {
    if data.len() != out.len() * wb || wb > (W::BITS / 8) as usize {
        return None;
    }
    for (o, c) in out.iter_mut().zip(data.chunks_exact(wb)) {
        let v = W::read_le(c);
        if v >= q {
            return None;
        }
        *o = v;
    }
    Some(())
}

/// Values as wb-byte little-endian words.
pub fn write_values<W: Word>(vals: &[W], wb: usize, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + vals.len() * wb, 0);
    for (&v, o) in vals.iter().zip(out[start..].chunks_exact_mut(wb)) {
        v.write_le(o);
    }
}

/// A format-2 ciphertext: (k, ct0 (eta rows of k values), ct1 (n values)),
/// or None unless the header matches (q, eta, n), 1 <= k <= n, the length is
/// exact and every value is below q.
pub fn read_ciphertext<W: Word>(data: &[u8], q: W, eta: usize, logn: u32) -> Option<(usize, Vec<W>, Vec<W>)> {
    let wb = value_bytes(q.bits());
    let n = 1usize << logn;
    let (hwb, heta, hlogn, k) = header(data)?;
    if hwb != wb || heta != eta || hlogn != logn || k < 1 || k > n
        || data.len() != BIN2_HEADER + (eta * k + n) * wb
    {
        return None;
    }
    let mut ct0 = vec![W::ZERO; eta * k];
    let mut ct1 = vec![W::ZERO; n];
    let split = BIN2_HEADER + eta * k * wb;
    read_strict(&data[BIN2_HEADER..split], q, wb, &mut ct0)?;
    read_strict(&data[split..], q, wb, &mut ct1)?;
    Some((k, ct0, ct1))
}

/// Format 2 of (k, ct0 rows truncated to k, ct1): "PFRL", 2, wb, eta (u16 LE),
/// log2 n, 0, k (u32 LE); then eta rows of k values and n values of ct1.
pub fn write_ciphertext<W: Word>(qbits: u32, logn: u32, eta: usize, k: usize, ct0: &[W], ct1: &[W], out: &mut Vec<u8>) {
    let wb = value_bytes(qbits);
    let n = 1usize << logn;
    out.extend_from_slice(BIN2_MAGIC);
    out.extend_from_slice(&[BIN2_VERSION, wb as u8, (eta & 0xFF) as u8, (eta >> 8) as u8, logn as u8, 0]);
    out.extend_from_slice(&(k as u32).to_le_bytes());
    for r in 0..eta {
        write_values(&ct0[r * n..r * n + k], wb, out);
    }
    write_values(ct1, wb, out);
}

/// ThresholdMCFERingLWE.share_decrypt of one chunk for ct0 rows of k values
/// (format 2) once its noise is drawn: w = lambda (Delta z + sum_s ct1_s sk_s)
/// + e and c_y = sum_s sum_row y_{s,row} ct0_{s,row} mod q, both for the first
/// k coefficients (w.len() = c_y.len() = noise.len() = k). `ct0[s]` holds
/// eta rows of k values, `ct1[s]` n values, `ym[s eta + r]` the weights and
/// `sk_hats[s]` NTT(sk_s share) in Montgomery form (ring.pw_mul_add's form).
#[allow(clippy::too_many_arguments)]
pub fn share_partial_k<W: Word>(
    ring: &Ring<W>, eta: usize, ct0: &[&[W]], ct1: &[&[W]], ym: &[W], sk_hats: &[&[W]], z: &[W], delta: W,
    lambda: W, noise: &[W], w: &mut [W], c_y: &mut [W],
) {
    let n = ring.n;
    let k = w.len();
    let m = &ring.m;
    let (mut acc, mut xh, mut kp) = (vec![W::ZERO; n], vec![W::ZERO; n], vec![W::ZERO; n]);
    for (c1, skh) in ct1.iter().zip(sk_hats) {
        ring.forward(c1, &mut xh);
        ring.pw_mul_add(&mut acc, skh, &xh);
    }
    ring.inverse(&acc, &mut kp);
    let (dm, lm) = (m.to_mont(delta), m.to_mont(lambda));
    for i in 0..k {
        let key = m.add(m.mul(z[i], dm), kp[i]);
        w[i] = m.add(m.mul(key, lm), noise[i]);
        let mut cy = W::ZERO;
        for (s, c0) in ct0.iter().enumerate() {
            for r in 0..eta {
                cy = m.add(cy, m.mul(c0[r * k + i], ym[s * eta + r]));
            }
        }
        c_y[i] = cy;
    }
    // the key part determines the key share's products with the ciphertexts
    wipe_words(&mut acc);
    wipe_words(&mut kp);
}

/// ThresholdMCFERingLWE._combine_group on values: decode_scaled of c_y - sum_k w_k
/// mod q (see rlwe::share_combine); false if a result does not fit an i64.
pub fn combine<W: Word>(m: &Modulus<W>, c_y: &[W], ws: &[&[W]], delta: W, bound: i128, out: &mut [i64], none: &mut [u8]) -> bool {
    for i in 0..out.len() {
        let v = ws.iter().fold(c_y[i], |acc, w| m.sub(acc, w[i]));
        match decode_scaled_w(v, m.q, delta, bound) {
            Ok(Some(mu)) => {
                out[i] = mu;
                none[i] = 0;
            }
            Ok(None) => {
                out[i] = 0;
                none[i] = 1;
            }
            Err(()) => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::U256;

    /// SHAKE-256 test vector (FIPS 202, empty message) and the Python formula.
    #[test]
    fn uniform_flooding_matches_the_formula() {
        let q = U256::ONE.shl(200).wadd(U256::from_u128(0x2_0001)); // odd
        let m = Modulus::new(q);
        let material = b"pyfe4ai/tmcfe-ring-lwe/uniform-flooding/v1\x00\x00\x00\x00\x00\x00\x00\x03abc";
        for bits in [1u32, 7, 8, 9, 63, 64, 130, 198, 200] {
            let mut out = vec![U256::ZERO; 37];
            uniform_flooding(&m, bits, material, &mut out).unwrap();
            let nb = (bits as usize + 8) / 8;
            let mut stream = vec![0u8; nb * 37];
            shake256(material, &mut stream);
            for (i, &v) in out.iter().enumerate() {
                // the low bits + 1 bits of the big-endian block, minus 2^bits, mod q
                let mut be = [0u8; 32];
                be[32 - nb..].copy_from_slice(&stream[i * nb..(i + 1) * nb]);
                let full = U256::read_be(&be);
                let mut low = U256::ZERO;
                for b in 0..=bits {
                    if full.bit(b) {
                        low = low.wadd(U256::ONE.shl(b));
                    }
                }
                let off = U256::ONE.shl(bits);
                let want = if low >= off { low.wsub(off) } else { q.wsub(off.wsub(low)) };
                assert_eq!(v, want, "bits {} index {}", bits, i);
            }
        }
        let mut out = vec![U256::ZERO; 2];
        assert!(uniform_flooding(&m, 200, material, &mut out).is_ok()); // 2^200 < q
        assert!(uniform_flooding(&m, 201, material, &mut out).is_err());
        assert!(uniform_flooding(&m, 0, material, &mut out).is_err());
    }

    /// share_partial_k on an RNS ring of four primes: the schoolbook values mod q.
    #[test]
    fn partial_on_an_rns_ring() {
        use crate::arith::tests::{slow_mulmod, Lcg};
        use crate::ntt::tests::rns_ring;
        use crate::with_ring;
        let (n, eta, k) = (16usize, 2usize, 5usize);
        let (ring, q) = rns_ring(54, n, 4);
        let mut rng = Lcg(0xfeed);
        let poly = |rng: &mut Lcg| (0..n).map(|_| rng.below(q)).collect::<Vec<U256>>();
        let (ct0s, ct1s, sks): (Vec<_>, Vec<_>, Vec<_>) = (0..3)
            .map(|_| ((0..eta).flat_map(|_| poly(&mut rng)).collect::<Vec<U256>>(), poly(&mut rng), poly(&mut rng)))
            .fold((vec![], vec![], vec![]), |mut acc, (a, b, c)| { acc.0.push(a); acc.1.push(b); acc.2.push(c); acc });
        let ys: Vec<U256> = (0..3 * eta).map(|i| U256::from_u128(i as u128 + 2)).collect();
        let (z, noise) = (poly(&mut rng), poly(&mut rng));
        let (delta, lambda) = (U256::from_u128(12345), q.wsub(U256::from_u128(7)));
        let m = Modulus::<U256>::new(q);
        let mut want_w = vec![U256::ZERO; k];
        let mut key: Vec<U256> = z.iter().map(|&v| slow_mulmod(v, delta, q)).collect();
        for s in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    let p = slow_mulmod(ct1s[s][i], sks[s][j], q);
                    if i + j < n { key[i + j] = m.add(key[i + j], p) } else { key[i + j - n] = m.sub(key[i + j - n], p) }
                }
            }
        }
        let mut want_cy = vec![U256::ZERO; k];
        for i in 0..k {
            want_w[i] = m.add(slow_mulmod(key[i], lambda, q), noise[i]);
            for s in 0..3 {
                for r in 0..eta {
                    want_cy[i] = m.add(want_cy[i], slow_mulmod(ct0s[s][r * n + i], ys[s * eta + r], q));
                }
            }
        }
        with_ring!(&ring, r => {
            let cv = |v: &[U256]| v.iter().map(|&x| Word::from_u256(x)).collect::<Vec<_>>();
            let ct0k: Vec<Vec<_>> = ct0s.iter().map(|c| (0..eta).flat_map(|row| cv(&c[row * n..row * n + k])).collect()).collect();
            let ct1w: Vec<Vec<_>> = ct1s.iter().map(|c| cv(c)).collect();
            let hats: Vec<Vec<_>> = sks.iter().map(|c| { let mut h = cv(c); r.forward_mont(&cv(c), &mut h); h }).collect();
            let ym: Vec<_> = ys.iter().map(|&y| r.m.to_mont(Word::from_u256(y))).collect();
            let (mut w, mut cy) = (vec![Word::ZERO; k], vec![Word::ZERO; k]);
            let c0: Vec<&[_]> = ct0k.iter().map(|v| v.as_slice()).collect();
            let c1: Vec<&[_]> = ct1w.iter().map(|v| v.as_slice()).collect();
            let hr: Vec<&[_]> = hats.iter().map(|v| v.as_slice()).collect();
            share_partial_k(r, eta, &c0, &c1, &ym, &hr, &cv(&z), Word::from_u256(delta), Word::from_u256(lambda),
                            &cv(&noise[..k]), &mut w, &mut cy);
            assert_eq!(w.iter().map(|v| v.to_u256()).collect::<Vec<_>>(), want_w);
            assert_eq!(cy.iter().map(|v| v.to_u256()).collect::<Vec<_>>(), want_cy);
        });
    }

    #[test]
    fn format_2_round_trip_and_strictness() {
        let q: u128 = (1 << 100) + 0x3_0001;
        let (eta, logn, k) = (2usize, 3u32, 5usize);
        let n = 1 << logn;
        let ct0: Vec<u128> = (0..eta * n).map(|i| (i as u128 * 0x1234_5678_9ABC) % q).collect();
        let ct1: Vec<u128> = (0..n).map(|i| q - 1 - i as u128).collect();
        let mut data = Vec::new();
        write_ciphertext(128 - q.leading_zeros(), logn, eta, k, &ct0, &ct1, &mut data);
        assert_eq!(data.len(), BIN2_HEADER + (eta * k + n) * 13);
        let (kk, c0, c1) = read_ciphertext(&data, q, eta, logn).unwrap();
        assert_eq!(kk, k);
        assert_eq!(c0, [&ct0[..k], &ct0[n..n + k]].concat());
        assert_eq!(c1, ct1);
        // a value q (another encoding of 0), a wrong k, length or version: refused
        let mut bad = data.clone();
        q.write_le(&mut bad[BIN2_HEADER..BIN2_HEADER + 13]);
        assert!(read_ciphertext(&bad, q, eta, logn).is_none());
        for (pos, v) in [(4usize, 1u8), (10, 0), (10, 9), (8, 4), (6, 1)] {
            let mut bad = data.clone();
            bad[pos] = v;
            assert!(read_ciphertext(&bad, q, eta, logn).is_none(), "byte {} = {}", pos, v);
        }
        assert!(read_ciphertext(&data[..data.len() - 1], q, eta, logn).is_none());
    }
}
