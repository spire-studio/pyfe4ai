//! Shamir sharing as pyfe4ai.schemes.threshold.sharing.share_secret computes
//! it: the t - 1 random coefficients of each secret drawn as
//! random.SystemRandom.randrange(0, q) draws them -- getrandbits(bits(q)) from
//! ceil(bits(q) / 8) big-endian bytes, drawn again while >= q -- and the
//! polynomial evaluated at every point.

use crate::arith::{Modulus, Word};

/// Uniform below q as SystemRandom._randbelow(q), reading bytes from `fill`
/// (None when it has no more).
pub fn below<W: Word, F: FnMut(&mut [u8]) -> bool>(q: W, buf: &mut [u8], fill: &mut F) -> Option<W> {
    let k = q.bits();
    let excess = 8 * buf.len() as u32 - k;
    loop {
        if !fill(buf) {
            return None;
        }
        let mut r = W::read_be(buf);
        for _ in 0..excess {
            r = r.shr1();
        }
        if r < q {
            return Some(r);
        }
    }
}

/// Shares of every secret (reduced mod q) for a threshold t at the points:
/// out[p][i] = f_i(points[p]), f_i(0) = secrets[i]; coefficients drawn secret by
/// secret, t - 1 each. None if `fill` runs out of bytes.
pub fn share<W: Word, F: FnMut(&mut [u8]) -> bool>(
    m: &Modulus<W>, secrets: &[W], t: usize, points: &[u64], mut fill: F,
) -> Option<Vec<Vec<W>>> {
    let nb = ((m.q.bits() + 7) / 8) as usize;
    let xm: Vec<W> = points.iter().map(|&x| m.to_mont(m.reduce_u128(x as u128))).collect();
    let mut out: Vec<Vec<W>> = points.iter().map(|_| Vec::with_capacity(secrets.len())).collect();
    let mut buf = vec![0u8; nb];
    let mut coeffs = vec![W::ZERO; t.max(1)];
    let mut ok = true;
    'secrets: for &s in secrets {
        coeffs[0] = s;
        for j in 1..coeffs.len() {
            match below(m.q, &mut buf, &mut fill) {
                Some(c) => coeffs[j] = c,
                None => {
                    ok = false;
                    break 'secrets;
                }
            }
        }
        for (p, &x) in xm.iter().enumerate() {
            // Horner: the same value mod q as sum_j c_j x^j
            let acc = coeffs.iter().rev().fold(W::ZERO, |acc, &c| m.add(m.mul(acc, x), c));
            out[p].push(acc);
        }
    }
    for c in coeffs.iter_mut() {
        unsafe { std::ptr::write_volatile(c, W::ZERO) };
    }
    crate::rlwe::wipe_bytes(&mut buf);
    if ok {
        return Some(out);
    }
    for row in out.iter_mut() {
        crate::rlwe::wipe_words(row); // the shares of the secrets dealt before the bytes ran out
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::U256;

    #[test]
    fn shares_reconstruct_and_draws_reject() {
        let q = (1u128 << 100) + 277; // 101 bits: 13 bytes, 3 excess bits (no prime needed here)
        let m = Modulus::<u128>::new(q);
        let secrets = [5u128, q - 1, 0];
        let mut stream = (0u8..=255).cycle();
        let shares = share(&m, &secrets, 3, &[1, 2, 3], |b: &mut [u8]| {
            b.iter_mut().for_each(|x| *x = stream.next().unwrap());
            true
        })
        .unwrap();
        // Lagrange at 0 over the points 1, 2, 3: 3 f(1) - 3 f(2) + f(3)
        for i in 0..3 {
            let v = m.add(m.sub(m.mulp(3, shares[0][i]), m.mulp(3, shares[1][i])), shares[2][i]);
            assert_eq!(v, secrets[i]);
        }
        // the first draw: bytes 0..13 big-endian, shifted right by 3
        let mut buf = [0u8; 13];
        let mut s2 = (0u8..=255).cycle();
        let r = below(q, &mut buf, &mut |b: &mut [u8]| {
            b.iter_mut().for_each(|x| *x = s2.next().unwrap());
            true
        });
        let want = (0u8..13).fold(0u128, |v, x| (v << 8) | x as u128) >> 3;
        assert_eq!(r, Some(want));
        // all-ones bytes (2^101 - 1 >= q) are drawn again
        let q2 = U256::from_u128((1u128 << 100) + 1);
        let mut n = 0;
        let r = below(q2, &mut [0u8; 13], &mut |b: &mut [u8]| {
            n += 1;
            b.iter_mut().for_each(|x| *x = if n == 1 { 0xff } else { 0 });
            true
        });
        assert_eq!((r, n), (Some(U256::ZERO), 2));
        assert!(share(&m, &secrets, 2, &[1], |_: &mut [u8]| false).is_none());
    }
}
