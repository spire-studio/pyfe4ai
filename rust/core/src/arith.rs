//! Modular arithmetic on u64 words (q < 2^62) and u128 words (q < 2^125):
//! Montgomery multiplication, Shoup multiplication by precomputed constants.

/// A machine word of the arithmetic.
pub trait Word:
    Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Send + Sync + core::fmt::Debug + 'static
{
    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;
    fn from_u128(v: u128) -> Self;
    fn to_u128(self) -> u128;
    fn wadd(self, o: Self) -> Self;
    fn wsub(self, o: Self) -> Self;
    /// Low BITS of the product.
    fn wmul(self, o: Self) -> Self;
    /// (high, low) BITS of the full product.
    fn full(self, o: Self) -> (Self, Self);
    fn shl1(self) -> Self;
    fn bit0(self) -> bool;
}

impl Word for u64 {
    const BITS: u32 = 64;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    #[inline(always)]
    fn from_u128(v: u128) -> Self {
        v as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn wadd(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline(always)]
    fn wsub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline(always)]
    fn wmul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline(always)]
    fn full(self, o: Self) -> (Self, Self) {
        let t = (self as u128) * (o as u128);
        ((t >> 64) as u64, t as u64)
    }
    #[inline(always)]
    fn shl1(self) -> Self {
        self << 1
    }
    #[inline(always)]
    fn bit0(self) -> bool {
        self & 1 == 1
    }
}

impl Word for u128 {
    const BITS: u32 = 128;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    #[inline(always)]
    fn from_u128(v: u128) -> Self {
        v
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self
    }
    #[inline(always)]
    fn wadd(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline(always)]
    fn wsub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline(always)]
    fn wmul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline(always)]
    fn full(self, o: Self) -> (Self, Self) {
        let (a0, a1) = (self as u64, (self >> 64) as u64);
        let (b0, b1) = (o as u64, (o >> 64) as u64);
        let p00 = (a0 as u128) * (b0 as u128);
        let p01 = (a0 as u128) * (b1 as u128);
        let p10 = (a1 as u128) * (b0 as u128);
        let p11 = (a1 as u128) * (b1 as u128);
        let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
        let lo = (p00 as u64 as u128) | (mid << 64);
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        (hi, lo)
    }
    #[inline(always)]
    fn shl1(self) -> Self {
        self << 1
    }
    #[inline(always)]
    fn bit0(self) -> bool {
        self & 1 == 1
    }
}

/// Arithmetic modulo an odd q, in Montgomery form with R = 2^BITS.
#[derive(Clone, Debug)]
pub struct Modulus<W: Word> {
    pub q: W,
    /// -q^-1 mod 2^BITS.
    pub qinv: W,
    /// R mod q.
    pub rmod: W,
    /// R^2 mod q.
    pub r2: W,
    pub bits: u32,
}

impl<W: Word> Modulus<W> {
    /// Needs q odd and q < 2^62 (u64) or q < 2^125 (u128).
    pub fn new(q: u128) -> Self {
        let qw = W::from_u128(q);
        let mut inv = qw; // q q = 1 mod 8
        for _ in 0..7 {
            inv = inv.wmul(W::from_u128(2).wsub(qw.wmul(inv)));
        }
        let rmod = if W::BITS == 64 {
            W::from_u128((1u128 << 64) % q)
        } else {
            W::from_u128((u128::MAX % q + 1) % q)
        };
        let mut m = Modulus { q: qw, qinv: W::ZERO.wsub(inv), rmod, r2: rmod, bits: 128 - q.leading_zeros() };
        let mut r2 = rmod;
        for _ in 0..W::BITS {
            r2 = m.add(r2, r2); // R 2^BITS = R^2 mod q
        }
        m.r2 = r2;
        m
    }
    #[inline(always)]
    pub fn add(&self, a: W, b: W) -> W {
        let s = a.wadd(b);
        if s >= self.q { s.wsub(self.q) } else { s }
    }
    #[inline(always)]
    pub fn sub(&self, a: W, b: W) -> W {
        if a >= b { a.wsub(b) } else { a.wadd(self.q).wsub(b) }
    }
    /// Montgomery product a b R^-1 mod q (a < R, b < q), in [0, q).
    #[inline(always)]
    pub fn mul(&self, a: W, b: W) -> W {
        let (hi, lo) = a.full(b);
        let m = lo.wmul(self.qinv);
        let (mh, _) = m.full(self.q);
        // lo + (m q mod R) is 0 or exactly R
        let r = hi.wadd(mh).wadd(if lo != W::ZERO { W::ONE } else { W::ZERO });
        if r >= self.q { r.wsub(self.q) } else { r }
    }
    #[inline(always)]
    pub fn to_mont(&self, x: W) -> W {
        self.mul(x, self.r2)
    }
    #[inline(always)]
    pub fn from_mont(&self, x: W) -> W {
        self.mul(x, W::ONE)
    }
    /// Plain a b mod q.
    #[inline(always)]
    pub fn mulp(&self, a: W, b: W) -> W {
        self.to_mont(self.mul(a, b))
    }
    /// Plain b^e mod q.
    pub fn pow(&self, b: W, mut e: u128) -> W {
        let mut r = self.rmod;
        let mut x = self.to_mont(b);
        while e != 0 {
            if e & 1 == 1 {
                r = self.mul(r, x);
            }
            x = self.mul(x, x);
            e >>= 1;
        }
        self.from_mont(r)
    }
    /// floor(w 2^BITS / q) for w < q.
    pub fn shoup_quot(&self, w: W) -> W {
        if W::BITS == 64 {
            return W::from_u128((w.to_u128() << 64) / self.q.to_u128());
        }
        let q = self.q.to_u128();
        let (mut r, mut quot) = (w.to_u128(), 0u128);
        for _ in 0..128 {
            r <<= 1;
            quot <<= 1;
            if r >= q {
                r -= q;
                quot |= 1;
            }
        }
        W::from_u128(quot)
    }
    /// x w mod q in [0, 2q) for any word x, with wp = shoup_quot(w).
    #[inline(always)]
    pub fn shoup(&self, x: W, w: W, wp: W) -> W {
        let (hi, _) = x.full(wp);
        x.wmul(w).wsub(hi.wmul(self.q))
    }
    /// v mod q for a signed 128-bit integer.
    #[inline(always)]
    pub fn reduce_i128(&self, v: i128) -> W {
        W::from_u128(reduce_i128(v, self.q.to_u128()))
    }
    /// v mod q for an unsigned 128-bit integer.
    #[inline(always)]
    pub fn reduce_u128(&self, v: u128) -> W {
        let q = self.q.to_u128();
        W::from_u128(if v < q { v } else { v % q })
    }
}

/// v mod q in [0, q).
#[inline(always)]
pub fn reduce_i128(v: i128, q: u128) -> u128 {
    if v >= 0 {
        let u = v as u128;
        return if u < q { u } else { u % q };
    }
    let mut m = v.unsigned_abs();
    if m >= q {
        m %= q;
    }
    if m == 0 { 0 } else { q - m }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slow_mulmod(a: u128, b: u128, q: u128) -> u128 {
        let (mut r, mut a, mut b) = (0u128, a % q, b);
        while b != 0 {
            if b & 1 == 1 {
                r = (r + a) % q;
            }
            a = (a << 1) % q;
            b >>= 1;
        }
        r
    }

    #[test]
    fn montgomery_and_shoup_match_slow_products() {
        let qs: [u128; 4] = [97, (1 << 61) - 1, 0x1_0000_0000_0000_0000_4001, (1u128 << 124) + 0x9001];
        let mut seed = 0x1234_5678_9abc_def0_u128;
        let mut next = |q: u128| {
            seed = seed.wrapping_mul(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f).wrapping_add(0x9e37_79b9);
            (seed >> 3) % q
        };
        for &q in &qs {
            for _ in 0..200 {
                let (a, b) = (next(q), next(q));
                let want = slow_mulmod(a, b, q);
                if q < (1 << 62) {
                    let m = Modulus::<u64>::new(q);
                    assert_eq!(m.mulp(a as u64, b as u64) as u128, want);
                    let s = m.shoup(a as u64, b as u64, m.shoup_quot(b as u64)) as u128;
                    assert_eq!(s % q, want);
                    assert!(s < 2 * q);
                }
                let m = Modulus::<u128>::new(q);
                assert_eq!(m.mulp(a, b), want);
                let s = m.shoup(a, b, m.shoup_quot(b));
                assert_eq!(s % q, want);
                assert!(s < 2 * q);
            }
        }
        assert_eq!(reduce_i128(-5, 7), 2);
        assert_eq!(reduce_i128(i128::MIN, 7), (7 - (i128::MIN.unsigned_abs() % 7)) % 7);
    }
}
