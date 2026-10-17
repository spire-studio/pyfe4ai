//! Modular arithmetic on u64 words (q < 2^62), u128 words (q < 2^125) and
//! 256-bit words (q < 2^253): Montgomery multiplication, Shoup multiplication
//! by precomputed constants.

use core::cmp::Ordering;

/// A machine word of the arithmetic.
pub trait Word:
    Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Send + Sync + core::fmt::Debug + 'static
{
    const BITS: u32;
    /// Largest bit length of q (Harvey's lazy butterflies need 4q < 2^BITS).
    const MAX_Q_BITS: u32;
    /// Longest decimal string `parse_dec` reads.
    const DEC_DIGITS: usize;
    const ZERO: Self;
    const ONE: Self;
    /// v, which must fit the word.
    fn from_u128(v: u128) -> Self;
    /// The low 128 bits.
    fn to_u128(self) -> u128;
    /// The low BITS of v.
    fn from_u256(v: U256) -> Self;
    fn to_u256(self) -> U256;
    fn wadd(self, o: Self) -> Self;
    fn wsub(self, o: Self) -> Self;
    /// Low BITS of the product.
    fn wmul(self, o: Self) -> Self;
    /// (high, low) BITS of the full product.
    fn full(self, o: Self) -> (Self, Self);
    fn shl1(self) -> Self;
    fn shr1(self) -> Self;
    /// self << s for s < BITS.
    fn shl(self, s: u32) -> Self;
    fn bit0(self) -> bool;
    /// Bit i (i < BITS).
    fn bit(self, i: u32) -> bool;
    /// Bit length (0 for 0).
    fn bits(self) -> u32;
    /// self mod d for d > 0.
    fn rem(self, d: Self) -> Self;
    /// self mod d for d > 0.
    fn rem_u64(self, d: u64) -> u64;
    /// From little-endian bytes (at most BITS / 8; missing high bytes are 0).
    fn read_le(b: &[u8]) -> Self;
    /// The low out.len() bytes, little-endian (out.len() <= BITS / 8).
    fn write_le(self, out: &mut [u8]);
    /// From big-endian bytes (at most BITS / 8).
    fn read_be(b: &[u8]) -> Self;
    /// int(s) % q for an optional '-' and 1..DEC_DIGITS ASCII digits; None otherwise.
    fn parse_dec(s: &[u8], q: Self) -> Option<Self>;
    /// Decimal digits appended to `out`.
    fn format_dec(self, out: &mut Vec<u8>);
}

impl Word for u64 {
    const BITS: u32 = 64;
    const MAX_Q_BITS: u32 = 62;
    const DEC_DIGITS: usize = 38;
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
    fn from_u256(v: U256) -> Self {
        v.0[0]
    }
    #[inline(always)]
    fn to_u256(self) -> U256 {
        U256([self, 0, 0, 0])
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
    fn shr1(self) -> Self {
        self >> 1
    }
    #[inline(always)]
    fn shl(self, s: u32) -> Self {
        self << s
    }
    #[inline(always)]
    fn bit0(self) -> bool {
        self & 1 == 1
    }
    #[inline(always)]
    fn bit(self, i: u32) -> bool {
        (self >> i) & 1 == 1
    }
    #[inline(always)]
    fn bits(self) -> u32 {
        64 - self.leading_zeros()
    }
    #[inline(always)]
    fn rem(self, d: Self) -> Self {
        self % d
    }
    #[inline(always)]
    fn rem_u64(self, d: u64) -> u64 {
        self % d
    }
    fn read_le(b: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        buf[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(buf)
    }
    fn write_le(self, out: &mut [u8]) {
        let n = out.len();
        out.copy_from_slice(&self.to_le_bytes()[..n]);
    }
    fn read_be(b: &[u8]) -> Self {
        b.iter().fold(0u64, |v, &x| (v << 8) | x as u64)
    }
    #[inline]
    fn parse_dec(s: &[u8], q: Self) -> Option<Self> {
        crate::codec::parse_dec(s, q as u128).map(|v| v as u64)
    }
    #[inline]
    fn format_dec(self, out: &mut Vec<u8>) {
        crate::codec::format_dec(self as u128, out)
    }
}

impl Word for u128 {
    const BITS: u32 = 128;
    const MAX_Q_BITS: u32 = 125;
    const DEC_DIGITS: usize = 38;
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
    fn from_u256(v: U256) -> Self {
        v.0[0] as u128 | (v.0[1] as u128) << 64
    }
    #[inline(always)]
    fn to_u256(self) -> U256 {
        U256([self as u64, (self >> 64) as u64, 0, 0])
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
    fn shr1(self) -> Self {
        self >> 1
    }
    #[inline(always)]
    fn shl(self, s: u32) -> Self {
        self << s
    }
    #[inline(always)]
    fn bit0(self) -> bool {
        self & 1 == 1
    }
    #[inline(always)]
    fn bit(self, i: u32) -> bool {
        (self >> i) & 1 == 1
    }
    #[inline(always)]
    fn bits(self) -> u32 {
        128 - self.leading_zeros()
    }
    #[inline(always)]
    fn rem(self, d: Self) -> Self {
        self % d
    }
    #[inline(always)]
    fn rem_u64(self, d: u64) -> u64 {
        (self % d as u128) as u64
    }
    fn read_le(b: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        buf[..b.len()].copy_from_slice(b);
        u128::from_le_bytes(buf)
    }
    fn write_le(self, out: &mut [u8]) {
        let n = out.len();
        out.copy_from_slice(&self.to_le_bytes()[..n]);
    }
    fn read_be(b: &[u8]) -> Self {
        b.iter().fold(0u128, |v, &x| (v << 8) | x as u128)
    }
    #[inline]
    fn parse_dec(s: &[u8], q: Self) -> Option<Self> {
        crate::codec::parse_dec(s, q)
    }
    #[inline]
    fn format_dec(self, out: &mut Vec<u8>) {
        crate::codec::format_dec(self, out)
    }
}

/// An unsigned 256-bit integer: four 64-bit limbs, least significant first.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct U256(pub [u64; 4]);

impl core::fmt::Debug for U256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = Vec::new();
        self.format_dec(&mut s);
        f.write_str(core::str::from_utf8(&s).unwrap())
    }
}

impl Ord for U256 {
    #[inline(always)]
    fn cmp(&self, o: &Self) -> Ordering {
        let (a, b) = (&self.0, &o.0);
        (a[3], a[2], a[1], a[0]).cmp(&(b[3], b[2], b[1], b[0]))
    }
}

impl PartialOrd for U256 {
    #[inline(always)]
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[inline(always)]
fn adc(a: u64, b: u64, c: u64) -> (u64, u64) {
    let s = a as u128 + b as u128 + c as u128;
    (s as u64, (s >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let d = (a as u128).wrapping_sub(b as u128 + borrow as u128);
    (d as u64, (d >> 127) as u64)
}

/// 10^19, the largest power of ten below 2^64 (its top bit is set).
const TEN19: u64 = 10_000_000_000_000_000_000;
/// floor((2^128 - 1) / TEN19) - 2^64, the reciprocal of Moller and Granlund.
const TEN19_INV: u64 = ((u128::MAX / TEN19 as u128) - (1u128 << 64)) as u64;

/// (u1 2^64 + u0) / d and the remainder for a normalised d and u1 < d
/// (Moller, Granlund, "Improved division by invariant integers", Alg. 4).
#[inline(always)]
fn div2by1(u1: u64, u0: u64, d: u64, v: u64) -> (u64, u64) {
    let p = (v as u128) * (u1 as u128);
    let qq = p.wrapping_add(((u1 as u128) << 64) | u0 as u128);
    let mut q1 = ((qq >> 64) as u64).wrapping_add(1);
    let q0 = qq as u64;
    let mut r = u0.wrapping_sub(q1.wrapping_mul(d));
    if r > q0 {
        q1 = q1.wrapping_sub(1);
        r = r.wrapping_add(d);
    }
    if r >= d {
        q1 += 1;
        r -= d;
    }
    (q1, r)
}

impl U256 {
    pub const MAX: U256 = U256([u64::MAX; 4]);

    pub fn from_le_bytes(b: &[u8; 32]) -> Self {
        let mut l = [0u64; 4];
        for (i, c) in b.chunks_exact(8).enumerate() {
            l[i] = u64::from_le_bytes(c.try_into().unwrap());
        }
        U256(l)
    }

    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut b = [0u8; 32];
        for i in 0..4 {
            b[8 * i..8 * i + 8].copy_from_slice(&self.0[i].to_le_bytes());
        }
        b
    }

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    /// self m + a and the carry limb.
    #[inline(always)]
    pub fn mul_small_add(self, m: u64, a: u64) -> (Self, u64) {
        let mut r = [0u64; 4];
        let mut c = a;
        for i in 0..4 {
            let t = (self.0[i] as u128) * (m as u128) + c as u128;
            r[i] = t as u64;
            c = (t >> 64) as u64;
        }
        (U256(r), c)
    }

    /// (self / 10^19, self mod 10^19).
    #[inline(always)]
    fn divrem_ten19(self) -> (Self, u64) {
        let mut q = [0u64; 4];
        let mut r = 0u64;
        for i in (0..4).rev() {
            let (qi, ri) = div2by1(r, self.0[i], TEN19, TEN19_INV);
            q[i] = qi;
            r = ri;
        }
        (U256(q), r)
    }
}

impl Word for U256 {
    const BITS: u32 = 256;
    const MAX_Q_BITS: u32 = 253;
    const DEC_DIGITS: usize = 78; // 2^256 < 10^78
    const ZERO: Self = U256([0; 4]);
    const ONE: Self = U256([1, 0, 0, 0]);
    #[inline(always)]
    fn from_u128(v: u128) -> Self {
        U256([v as u64, (v >> 64) as u64, 0, 0])
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self.0[0] as u128 | (self.0[1] as u128) << 64
    }
    #[inline(always)]
    fn from_u256(v: U256) -> Self {
        v
    }
    #[inline(always)]
    fn to_u256(self) -> U256 {
        self
    }
    #[inline(always)]
    fn wadd(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        let (r0, c) = adc(a[0], b[0], 0);
        let (r1, c) = adc(a[1], b[1], c);
        let (r2, c) = adc(a[2], b[2], c);
        let (r3, _) = adc(a[3], b[3], c);
        U256([r0, r1, r2, r3])
    }
    #[inline(always)]
    fn wsub(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        let (r0, c) = sbb(a[0], b[0], 0);
        let (r1, c) = sbb(a[1], b[1], c);
        let (r2, c) = sbb(a[2], b[2], c);
        let (r3, _) = sbb(a[3], b[3], c);
        U256([r0, r1, r2, r3])
    }
    #[inline(always)]
    fn wmul(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        let mut r = [0u64; 4];
        for i in 0..4 {
            let mut c = 0u64;
            for j in 0..4 - i {
                let t = (a[i] as u128) * (b[j] as u128) + r[i + j] as u128 + c as u128;
                r[i + j] = t as u64;
                c = (t >> 64) as u64;
            }
        }
        U256(r)
    }
    #[inline(always)]
    fn full(self, o: Self) -> (Self, Self) {
        let (a, b) = (self.0, o.0);
        let mut r = [0u64; 8];
        for i in 0..4 {
            let mut c = 0u64;
            for j in 0..4 {
                let t = (a[i] as u128) * (b[j] as u128) + r[i + j] as u128 + c as u128;
                r[i + j] = t as u64;
                c = (t >> 64) as u64;
            }
            r[i + 4] = c;
        }
        (U256([r[4], r[5], r[6], r[7]]), U256([r[0], r[1], r[2], r[3]]))
    }
    #[inline(always)]
    fn shl1(self) -> Self {
        let a = self.0;
        U256([a[0] << 1, (a[1] << 1) | (a[0] >> 63), (a[2] << 1) | (a[1] >> 63), (a[3] << 1) | (a[2] >> 63)])
    }
    #[inline(always)]
    fn shr1(self) -> Self {
        let a = self.0;
        U256([(a[0] >> 1) | (a[1] << 63), (a[1] >> 1) | (a[2] << 63), (a[2] >> 1) | (a[3] << 63), a[3] >> 1])
    }
    fn shl(self, s: u32) -> Self {
        let (limbs, bits) = ((s / 64) as usize, s % 64);
        let mut r = [0u64; 4];
        for i in limbs..4 {
            let lo = self.0[i - limbs];
            r[i] = lo << bits;
            if bits != 0 && i > limbs {
                r[i] |= self.0[i - limbs - 1] >> (64 - bits);
            }
        }
        U256(r)
    }
    #[inline(always)]
    fn bit0(self) -> bool {
        self.0[0] & 1 == 1
    }
    #[inline(always)]
    fn bit(self, i: u32) -> bool {
        (self.0[(i / 64) as usize] >> (i % 64)) & 1 == 1
    }
    #[inline(always)]
    fn bits(self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return 64 * i as u32 + 64 - self.0[i].leading_zeros();
            }
        }
        0
    }
    fn rem(self, d: Self) -> Self {
        if self < d {
            return self;
        }
        let shift = self.bits() - d.bits();
        let mut ds = d.shl(shift);
        let mut r = self;
        for _ in 0..=shift {
            if r >= ds {
                r = r.wsub(ds);
            }
            ds = ds.shr1();
        }
        r
    }
    fn rem_u64(self, d: u64) -> u64 {
        let mut r = 0u128;
        for i in (0..4).rev() {
            r = ((r << 64) | self.0[i] as u128) % d as u128;
        }
        r as u64
    }
    fn read_le(b: &[u8]) -> Self {
        let mut buf = [0u8; 32];
        buf[..b.len()].copy_from_slice(b);
        U256::from_le_bytes(&buf)
    }
    fn write_le(self, out: &mut [u8]) {
        let n = out.len();
        out.copy_from_slice(&self.to_le_bytes()[..n]);
    }
    fn read_be(b: &[u8]) -> Self {
        let mut buf = [0u8; 32];
        for (i, &x) in b.iter().rev().enumerate() {
            buf[i] = x;
        }
        U256::from_le_bytes(&buf)
    }
    fn parse_dec(s: &[u8], q: Self) -> Option<Self> {
        let neg = !s.is_empty() && s[0] == b'-';
        let d = if neg { &s[1..] } else { s };
        if d.is_empty() || d.len() > Self::DEC_DIGITS {
            return None;
        }
        let mut v = U256::ZERO;
        let first = d.len() % 19;
        let mut pos = 0;
        let mut pw = if first == 0 { 19 } else { first };
        while pos < d.len() {
            let mut chunk = 0u64;
            let mut scale = 1u64;
            for &c in &d[pos..pos + pw] {
                let x = c.wrapping_sub(b'0');
                if x > 9 {
                    return None;
                }
                chunk = chunk * 10 + x as u64;
                scale *= 10;
            }
            let (nv, carry) = v.mul_small_add(scale, chunk);
            if carry != 0 {
                return None; // >= 2^256: Python's
            }
            v = nv;
            pos += pw;
            pw = 19;
        }
        let v = if v < q { v } else { v.rem(q) };
        Some(if neg && !v.is_zero() { q.wsub(v) } else { v })
    }
    fn format_dec(self, out: &mut Vec<u8>) {
        if self.0[2] == 0 && self.0[3] == 0 && self.0[1] >> 63 == 0 {
            return crate::codec::format_dec(self.to_u128(), out); // below 2^127
        }
        let mut parts = [0u64; 5];
        let (mut x, mut k) = (self, 0);
        while !x.is_zero() {
            let (qq, r) = x.divrem_ten19();
            parts[k] = r;
            k += 1;
            x = qq;
        }
        crate::codec::format_dec(parts[k - 1] as u128, out);
        for &p in parts[..k - 1].iter().rev() {
            crate::codec::format_dec19(p, out);
        }
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
    /// Needs q odd and q < 2^MAX_Q_BITS.
    pub fn new(q: W) -> Self {
        let mut inv = q; // q q = 1 mod 8; Newton's step doubles the correct low bits
        for _ in 0..7 {
            inv = inv.wmul(W::from_u128(2).wsub(q.wmul(inv)));
        }
        let mut m = Modulus { q, qinv: W::ZERO.wsub(inv), rmod: W::ZERO, r2: W::ZERO, bits: q.bits() };
        let mut r = W::ONE.rem(q);
        for _ in 0..W::BITS {
            r = m.add(r, r); // 2^BITS mod q
        }
        m.rmod = r;
        for _ in 0..W::BITS {
            r = m.add(r, r); // R 2^BITS = R^2 mod q
        }
        m.r2 = r;
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
    pub fn pow(&self, b: W, e: W) -> W {
        let x = self.to_mont(b);
        let mut r = self.rmod;
        for i in (0..e.bits()).rev() {
            r = self.mul(r, r);
            if e.bit(i) {
                r = self.mul(r, x);
            }
        }
        self.from_mont(r)
    }
    /// floor(w 2^BITS / q) for w < q: with t = w R mod q, w R = wp q + t, so
    /// wp = -t q^-1 mod R (wp < R).
    pub fn shoup_quot(&self, w: W) -> W {
        self.to_mont(w).wmul(self.qinv)
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
        let m = self.reduce_u128(v.unsigned_abs());
        if v >= 0 || m == W::ZERO { m } else { self.q.wsub(m) }
    }
    /// v mod q for an unsigned 128-bit integer.
    #[inline(always)]
    pub fn reduce_u128(&self, v: u128) -> W {
        if W::BITS <= 128 {
            let q = self.q.to_u128();
            return W::from_u128(if v < q { v } else { v % q });
        }
        self.reduce(W::from_u128(v))
    }
    /// v mod q for any word.
    #[inline(always)]
    pub fn reduce(&self, v: W) -> W {
        if v < self.q { v } else { v.rem(self.q) }
    }
    /// v mod q for any 256-bit integer.
    pub fn reduce_u256(&self, v: U256) -> W {
        let q = self.q.to_u256();
        W::from_u256(if v < q { v } else { v.rem(q) })
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
pub(crate) mod tests {
    use super::*;

    /// a b mod q by doubling (any width).
    pub fn slow_mulmod<W: Word>(a: W, b: W, q: W) -> W {
        let (mut r, mut a) = (W::ZERO, a.rem(q));
        let m = Modulus { q, qinv: W::ZERO, rmod: W::ZERO, r2: W::ZERO, bits: q.bits() };
        for i in 0..b.bits() {
            if b.bit(i) {
                r = m.add(r, a);
            }
            a = m.add(a, a);
        }
        r
    }

    /// A 256-bit value from a decimal string (test helper).
    pub fn u256(s: &str) -> U256 {
        s.bytes().fold(U256::ZERO, |v, c| {
            let (r, carry) = v.mul_small_add(10, (c - b'0') as u64);
            assert_eq!(carry, 0);
            r
        })
    }

    pub struct Lcg(pub u128);
    impl Lcg {
        pub fn next(&mut self) -> u128 {
            self.0 = self.0.wrapping_mul(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f).wrapping_add(0x9e37_79b9);
            self.0
        }
        pub fn below(&mut self, q: U256) -> U256 {
            let v = U256([self.next() as u64, (self.next() >> 64) as u64, self.next() as u64, (self.next() >> 64) as u64]);
            v.rem(q)
        }
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
                    let m = Modulus::<u64>::new(q as u64);
                    assert_eq!(m.mulp(a as u64, b as u64) as u128, want);
                    let s = m.shoup(a as u64, b as u64, m.shoup_quot(b as u64)) as u128;
                    assert_eq!(s % q, want);
                    assert!(s < 2 * q);
                    assert_eq!(m.shoup_quot(b as u64) as u128, ((b << 64) / q));
                }
                let m = Modulus::<u128>::new(q);
                assert_eq!(m.mulp(a, b), want);
                let s = m.shoup(a, b, m.shoup_quot(b));
                assert_eq!(s % q, want);
                assert!(s < 2 * q);
                // the 256-bit word computes the same products for these moduli
                let m4 = Modulus::<U256>::new(U256::from_u128(q));
                let (a4, b4) = (U256::from_u128(a), U256::from_u128(b));
                assert_eq!(m4.mulp(a4, b4), U256::from_u128(want));
                let s4 = m4.shoup(a4, b4, m4.shoup_quot(b4));
                assert_eq!(s4.rem(m4.q), U256::from_u128(want));
                assert!(s4 < m4.q.shl1());
            }
        }
        assert_eq!(reduce_i128(-5, 7), 2);
        assert_eq!(reduce_i128(i128::MIN, 7), (7 - (i128::MIN.unsigned_abs() % 7)) % 7);
        let m = Modulus::<U256>::new(U256::from_u128(7));
        assert_eq!(m.reduce_i128(-5), U256::from_u128(2));
        assert_eq!(m.reduce_i128(i128::MIN), U256::from_u128(reduce_i128(i128::MIN, 7)));
    }

    #[test]
    fn wide_moduli() {
        // the ristretto255 group order 2^252 + 27742317777372353535851937790883648493, the
        // NTT prime after 2^211 for ring_n 8192, the prime before 2^253
        let qs = [
            u256("7237005577332262213973186563042994240857116359379907606001950938285454250989"),
            u256("3291009114642412084309938365114701009965471731267159726697299969"),
            u256("14474011154664524427946373126085988481658748083205070504932198000989141204719"),
        ];
        let mut rng = Lcg(42);
        for &q in &qs {
            let m = Modulus::<U256>::new(q);
            assert_eq!(m.q.wmul(m.qinv.wsub(U256::ZERO)), U256::ZERO.wsub(U256::ONE)); // q (-q^-1) = -1
            assert_eq!(m.rmod, U256::ZERO.wsub(q).rem(q));
            for _ in 0..300 {
                let (a, b) = (rng.below(q), rng.below(q));
                let want = slow_mulmod(a, b, q);
                assert_eq!(m.mulp(a, b), want);
                assert_eq!(m.from_mont(m.mul(m.to_mont(a), m.to_mont(b))), want);
                let s = m.shoup(a, b, m.shoup_quot(b));
                assert!(s < q.shl1());
                assert_eq!(s.rem(q), want);
                // Shoup for any word x (the lazy butterflies' inputs in [0, 4q))
                let x = a.wadd(q).wadd(q).wadd(q);
                let s = m.shoup(x, b, m.shoup_quot(b));
                assert!(s < q.shl1());
                assert_eq!(s.rem(q), slow_mulmod(x, b, q));
                assert_eq!(m.add(a, b), slow_add(a, b, q));
                assert_eq!(m.sub(a, b), slow_add(a, q.wsub(b), q));
            }
            // Fermat: a^(q-1) = 1
            let a = rng.below(q);
            assert_eq!(m.pow(a, q.wsub(U256::ONE)), U256::ONE);
            assert_eq!(m.mulp(a, m.pow(a, q.wsub(U256::from_u128(2)))), U256::ONE);
            assert_eq!(m.reduce_i128(-3), q.wsub(U256::from_u128(3)));
            assert_eq!(m.reduce_u256(U256::MAX), U256::MAX.rem(q));
        }
    }

    fn slow_add(a: U256, b: U256, q: U256) -> U256 {
        let s = a.wadd(b);
        if s < a || s >= q { s.wsub(q) } else { s }
    }

    #[test]
    fn u256_basics() {
        let a = u256("115792089237316195423570985008687907853269984665640564039457584007913129639935");
        assert_eq!(a, U256::MAX);
        let mut s = Vec::new();
        a.format_dec(&mut s);
        assert_eq!(s, b"115792089237316195423570985008687907853269984665640564039457584007913129639935");
        let b = u256("340282366920938463463374607431768211456"); // 2^128
        assert_eq!(b, U256([0, 0, 1, 0]));
        assert_eq!(b.bits(), 129);
        assert_eq!(U256::ONE.shl(200).shr1(), U256::ONE.shl(199));
        assert_eq!(U256::ONE.shl(64).wsub(U256::ONE), U256([u64::MAX, 0, 0, 0]));
        assert_eq!(U256::MAX.wadd(U256::ONE), U256::ZERO);
        let (hi, lo) = U256::MAX.full(U256::MAX); // (2^256 - 1)^2 = 2^512 - 2^257 + 1
        assert_eq!((hi, lo), (U256::MAX.wsub(U256::ONE), U256::ONE));
        assert_eq!(U256::MAX.rem_u64(TEN19), (U256::MAX.rem(U256::from_u128(TEN19 as u128))).0[0]);
        // around the u128 formatter's limit 2^127 and other limb boundaries
        for (k, d) in [(127u32, -1i32), (127, 0), (127, 1), (128, -1), (128, 0), (192, -1), (192, 0), (255, 0)] {
            let x = if d < 0 { U256::ONE.shl(k).wsub(U256::ONE) } else { U256::ONE.shl(k).wadd(U256::from_u128(d as u128)) };
            let mut t = Vec::new();
            x.format_dec(&mut t);
            assert_eq!(u256(core::str::from_utf8(&t).unwrap()), x, "2^{} {:+}", k, d);
        }
        let mut t = Vec::new();
        u256("300000000000000000000000000000000012345").format_dec(&mut t);
        assert_eq!(t, b"300000000000000000000000000000000012345");
        let mut rng = Lcg(7);
        for i in 0..1500 {
            // full-width values and values of 100..160 bits
            let x = if i % 3 == 0 { rng.below(U256::MAX) } else { rng.below(U256::ONE.shl(100 + (i % 61) as u32)) };
            let mut t = Vec::new();
            x.format_dec(&mut t);
            assert_eq!(U256::parse_dec(&t, U256::MAX), Some(x));
            let (qq, r) = x.divrem_ten19();
            assert_eq!(qq.mul_small_add(TEN19, r), (x, 0));
            let mut be = x.to_le_bytes();
            be.reverse();
            assert_eq!(U256::read_be(&be), x);
            assert_eq!(U256::read_le(&x.to_le_bytes()[..20]), U256([x.0[0], x.0[1], x.0[2] & 0xffff_ffff, 0]));
        }
    }
}
