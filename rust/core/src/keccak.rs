//! SHAKE-256 (FIPS 202) and the reduction of crypto_utils.hash_to_zq_vector.

use crate::arith::{Modulus, Word};

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

/// Keccak-f[1600] on 25 little-endian lanes A[x + 5 y].
pub fn keccak_f(a: &mut [u64; 25]) {
    for rc in RC {
        let c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
        let c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
        let c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
        let c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
        let c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
        let d0 = c4 ^ c1.rotate_left(1);
        let d1 = c0 ^ c2.rotate_left(1);
        let d2 = c1 ^ c3.rotate_left(1);
        let d3 = c2 ^ c4.rotate_left(1);
        let d4 = c3 ^ c0.rotate_left(1);
        // theta, rho, pi: B[y, 2x + 3y] = rot(A[x, y] ^ D[x], r[x, y])
        let b = [
            a[0] ^ d0,
            (a[6] ^ d1).rotate_left(44),
            (a[12] ^ d2).rotate_left(43),
            (a[18] ^ d3).rotate_left(21),
            (a[24] ^ d4).rotate_left(14),
            (a[3] ^ d3).rotate_left(28),
            (a[9] ^ d4).rotate_left(20),
            (a[10] ^ d0).rotate_left(3),
            (a[16] ^ d1).rotate_left(45),
            (a[22] ^ d2).rotate_left(61),
            (a[1] ^ d1).rotate_left(1),
            (a[7] ^ d2).rotate_left(6),
            (a[13] ^ d3).rotate_left(25),
            (a[19] ^ d4).rotate_left(8),
            (a[20] ^ d0).rotate_left(18),
            (a[4] ^ d4).rotate_left(27),
            (a[5] ^ d0).rotate_left(36),
            (a[11] ^ d1).rotate_left(10),
            (a[17] ^ d2).rotate_left(15),
            (a[23] ^ d3).rotate_left(56),
            (a[2] ^ d2).rotate_left(62),
            (a[8] ^ d3).rotate_left(55),
            (a[14] ^ d4).rotate_left(39),
            (a[15] ^ d0).rotate_left(41),
            (a[21] ^ d1).rotate_left(2),
        ];
        for y in (0..25).step_by(5) {
            let (b0, b1, b2, b3, b4) = (b[y], b[y + 1], b[y + 2], b[y + 3], b[y + 4]);
            a[y] = b0 ^ (!b1 & b2);
            a[y + 1] = b1 ^ (!b2 & b3);
            a[y + 2] = b2 ^ (!b3 & b4);
            a[y + 3] = b3 ^ (!b4 & b0);
            a[y + 4] = b4 ^ (!b0 & b1);
        }
        a[0] ^= rc;
    }
}

const RATE: usize = 136;

/// SHAKE-256(msg) truncated to out.len() bytes. The state and the buffers are
/// wiped on return (msg may hold a secret seed).
pub fn shake256(msg: &[u8], out: &mut [u8]) {
    let mut a = [0u64; 25];
    let mut chunks = msg.chunks_exact(RATE);
    for blk in &mut chunks {
        for (i, lane) in blk.chunks_exact(8).enumerate() {
            a[i] ^= u64::from_le_bytes(lane.try_into().unwrap());
        }
        keccak_f(&mut a);
    }
    let rest = chunks.remainder();
    let mut last = [0u8; RATE];
    last[..rest.len()].copy_from_slice(rest);
    last[rest.len()] ^= 0x1F;
    last[RATE - 1] ^= 0x80;
    for (i, lane) in last.chunks_exact(8).enumerate() {
        a[i] ^= u64::from_le_bytes(lane.try_into().unwrap());
    }
    let mut written = 0;
    let mut block = [0u8; RATE];
    loop {
        keccak_f(&mut a);
        for i in 0..RATE / 8 {
            block[8 * i..8 * i + 8].copy_from_slice(&a[i].to_le_bytes());
        }
        let take = (out.len() - written).min(RATE);
        out[written..written + take].copy_from_slice(&block[..take]);
        written += take;
        if written == out.len() {
            break;
        }
    }
    for lane in a.iter_mut() {
        unsafe { std::ptr::write_volatile(lane, 0) };
    }
    crate::rlwe::wipe_bytes(&mut last);
    crate::rlwe::wipe_bytes(&mut block);
}

/// hash_to_zq_vector's block size: bits(q) + 128 bits, in bytes.
pub fn zq_block(qbits: u32) -> usize {
    ((qbits + 128 + 7) / 8) as usize
}

/// hash_to_zq_vector after the SHAKE call: big-endian block k (nb bytes) mod q, as
/// sum_k limb_k (2^(BITS k) mod q), one Montgomery product per word-sized limb.
pub fn zq_reduce<W: Word>(m: &Modulus<W>, stream: &[u8], nb: usize, out: &mut [W]) {
    let lb = (W::BITS / 8) as usize;
    let limbs = (nb + lb - 1) / lb;
    let first = if nb % lb != 0 { nb % lb } else { lb };
    let shift = m.rmod;
    let mut cm = Vec::with_capacity(limbs);
    let mut pw = W::ONE;
    for _ in 0..limbs {
        cm.push(m.to_mont(pw));
        pw = m.mulp(pw, shift);
    }
    for (b, o) in out.iter_mut().enumerate() {
        let s = &stream[b * nb..(b + 1) * nb];
        let mut acc = W::ZERO;
        let mut pos = 0;
        for k in (0..limbs).rev() {
            let take = if k == limbs - 1 { first } else { lb };
            let limb = W::read_be(&s[pos..pos + take]);
            pos += take;
            acc = m.add(acc, m.mul(limb, cm[k]));
        }
        *o = acc;
    }
}

/// hash_to_zq_vector(material, q, count) for the encoded material, as words;
/// the stream is wiped (the material may hold a secret seed).
pub fn hash_to_zq<W: Word>(m: &Modulus<W>, material: &[u8], out: &mut [W]) {
    let nb = zq_block(m.bits);
    let mut stream = vec![0u8; nb * out.len()];
    shake256(material, &mut stream);
    zq_reduce(m, &stream, nb, out);
    crate::rlwe::wipe_bytes(&mut stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    #[test]
    fn shake256_known_answers() {
        let mut out = [0u8; 32];
        shake256(b"", &mut out);
        assert_eq!(hex(&out), "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");
    }
}
