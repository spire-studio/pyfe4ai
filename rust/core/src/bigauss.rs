//! The rounded Gaussian of wide widths:
//! pyfe4ai.utils.sampling_utils.discrete_gaussian_bigint.
//!
//! The reference draws b1 = getrandbits(p), then b2 = getrandbits(p), with
//! p = bits(ceil(sigma)) + 128, and returns rint(z sig) (nearest, ties to even)
//! for z = sqrt(-2 log u1) cos(2 pi u2), u1 = (b1 + 1) / 2^p, u2 = b2 / 2^p and
//! sig = mpfr(sigma), every operation an MPFR one at p bits. Here
//! X = sig sqrt(-2 ln u1) cos(2 pi u2) is evaluated in fixed point with n
//! fraction words, n = max(1, ceil((s + 64) / 64)) for sig < 2^s, and the
//! sample is decided when X is far from every half-integer:
//!
//! * the reference's rint argument is within 2^-118 of X: u1 and u2 are exact,
//!   sig is used as is, and the dozen roundings (2 pi, log, sqrt, 2 pi u2, cos,
//!   two products) are each of relative size at most 2^-p, which gives an
//!   error below 17.5 sig sqrt(2 p ln 2) 2^-p < 2^-118 since sig <= 2^(p - 128)
//!   and p <= MAX_PREC;
//! * the error of the evaluation here is below 2^(s + 10 - 64 n) (the bounds in
//!   ulps of 2^-64n are next to each step below, and the tests check the bound
//!   against MPFR), so <= 2^-54;
//! * a sample is returned only when the approximation is more than
//!   2^(max(s + 16 - 64 n, -100) + 1) from every half-integer; then both values
//!   round to the same integer. Otherwise (probability at most about 2^-45,
//!   when s is just below a multiple of 64; about 2^-62 at s = 47) it is
//!   reported undecided, and the caller computes it with the reference formula
//!   from the same b1 and b2 -- so for a given bit source the samples are the
//!   reference's.
//!
//! Not constant-time (neither is the reference).

use std::sync::OnceLock;

/// Bits per uniform handled: getrandbits(p) for p <= MAX_PREC.
pub const MAX_PREC: u32 = 64 * IN as u32;
/// Words of b1, b2 and of sig's mantissa.
const IN: usize = 12;
/// Fraction words of the evaluation, at most.
pub const MAX_FRAC: usize = 8;
/// Widths handled: sig < 2^MAX_SIGMA_BITS (so that n <= MAX_FRAC).
pub const MAX_SIGMA_BITS: i64 = 64 * MAX_FRAC as i64 - 64;
/// Words of a sample's magnitude, at most (|X| < 33 sig < 2^(MAX_SIGMA_BITS + 6)).
pub const MAG: usize = MAX_FRAC;
/// Flags of `fill`: the sample is (-1)^flag |mag|, or undecided.
pub const POSITIVE: u8 = 0;
pub const NEGATIVE: u8 = 1;
pub const UNDECIDED: u8 = 2;

// ------------------------------------------------------------ fixed point

/// W 64-bit words, little-endian; the last is the integer part, the others the
/// fraction: value = sum a[i] 2^(64 (i - (W - 1))). Products and quotients are
/// truncated (error below one ulp, 2^-64(W-1)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Fx<const W: usize>([u64; W]);

/// Room for the product of two Fx<12> (the tables' type).
const PROD: usize = 24;

impl<const W: usize> Fx<W> {
    const ZERO: Self = Fx([0; W]);

    #[inline(always)]
    fn int(v: u64) -> Self {
        let mut a = [0u64; W];
        a[W - 1] = v;
        Fx(a)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        let mut r = [0u64; W];
        let mut c = 0u64;
        for i in 0..W {
            let (s1, c1) = self.0[i].overflowing_add(o.0[i]);
            let (s2, c2) = s1.overflowing_add(c);
            r[i] = s2;
            c = (c1 | c2) as u64;
        }
        debug_assert!(c == 0);
        Fx(r)
    }

    /// self - o for self >= o.
    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        let mut r = [0u64; W];
        let mut b = 0u64;
        for i in 0..W {
            let (d1, b1) = self.0[i].overflowing_sub(o.0[i]);
            let (d2, b2) = d1.overflowing_sub(b);
            r[i] = d2;
            b = (b1 | b2) as u64;
        }
        debug_assert!(b == 0);
        Fx(r)
    }

    #[inline(always)]
    fn lt(&self, o: &Self) -> bool {
        for i in (0..W).rev() {
            if self.0[i] != o.0[i] {
                return self.0[i] < o.0[i];
            }
        }
        false
    }

    /// The truncated product (the integer part must not overflow).
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        let mut p = [0u64; PROD];
        for i in 0..W {
            let a = self.0[i] as u128;
            let mut c = 0u128;
            for j in 0..W {
                let t = a * o.0[j] as u128 + p[i + j] as u128 + c;
                p[i + j] = t as u64;
                c = t >> 64;
            }
            p[i + W] = c as u64;
        }
        debug_assert!(p[2 * W - 1] == 0);
        let mut r = [0u64; W];
        r.copy_from_slice(&p[W - 1..2 * W - 1]);
        Fx(r)
    }

    /// The exact product by m (must not overflow).
    #[inline(always)]
    fn mul_u64(&self, m: u64) -> Self {
        let mut r = [0u64; W];
        let mut c = 0u128;
        for i in 0..W {
            let t = self.0[i] as u128 * m as u128 + c;
            r[i] = t as u64;
            c = t >> 64;
        }
        debug_assert!(c == 0);
        Fx(r)
    }

    /// The truncated quotient by d > 0.
    #[inline(always)]
    fn div_u64(&self, d: u64) -> Self {
        let mut r = [0u64; W];
        let mut rem = 0u128;
        for i in (0..W).rev() {
            let cur = (rem << 64) | self.0[i] as u128;
            r[i] = (cur / d as u128) as u64;
            rem = cur % d as u128;
        }
        Fx(r)
    }

    /// self / 2^s, truncated.
    #[inline(always)]
    fn shr(&self, s: u32) -> Self {
        let (w, b) = ((s / 64) as usize, s % 64);
        let mut r = [0u64; W];
        for i in 0..W {
            let lo = if i + w < W { self.0[i + w] } else { 0 };
            let hi = if i + w + 1 < W { self.0[i + w + 1] } else { 0 };
            r[i] = if b == 0 { lo } else { (lo >> b) | (hi << (64 - b)) };
        }
        Fx(r)
    }

    /// self 2^s (must not overflow).
    #[inline(always)]
    fn shl(&self, s: u32) -> Self {
        let (w, b) = ((s / 64) as usize, s % 64);
        let mut r = [0u64; W];
        for i in (0..W).rev() {
            let lo = if i >= w { self.0[i - w] } else { 0 };
            let lo2 = if i > w { self.0[i - w - 1] } else { 0 };
            r[i] = if b == 0 { lo } else { (lo << b) | (lo2 >> (64 - b)) };
        }
        Fx(r)
    }

    /// An approximation (relative error about 2^-52 for values above 2^-64).
    fn to_f64(&self) -> f64 {
        let mut v = 0.0f64;
        let mut scale = 1.0f64;
        for i in (0..W).rev().take(3) {
            v += self.0[i] as f64 * scale;
            scale *= 1.0 / 18446744073709551616.0;
        }
        v
    }

    /// v (finite, 0 < v < 2^63), truncated.
    fn from_f64(v: f64) -> Self {
        let bits = v.to_bits();
        let e = ((bits >> 52) & 0x7ff) as i64 - 1075;
        let m = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
        // integer of the fixed point: m 2^(e + 64 (W - 1))
        let mut x = [0u64; 2];
        x[0] = m;
        let mut r = [0u64; W];
        let shift = e + 64 * (W as i64 - 1);
        for (i, w) in r.iter_mut().enumerate() {
            *w = bits64(&x, 64 * i as i64 - shift);
        }
        Fx(r)
    }

    /// The top W words of a table entry.
    #[inline(always)]
    fn tab(t: &Tw) -> Self {
        let mut a = [0u64; W];
        a.copy_from_slice(&t[TW - W..]);
        Fx(a)
    }
}

/// Bits [p, p + 64) of the little-endian integer x (bits below 0 are 0).
#[inline(always)]
fn bits64(x: &[u64], p: i64) -> u64 {
    if p <= -64 {
        return 0;
    }
    if p < 0 {
        return x[0] << (-p) as u32;
    }
    let w = (p / 64) as usize;
    let s = (p % 64) as u32;
    let lo = if w < x.len() { x[w] } else { 0 };
    if s == 0 {
        return lo;
    }
    let hi = if w + 1 < x.len() { x[w + 1] } else { 0 };
    (lo >> s) | (hi << (64 - s))
}

/// floor(x / 2^lo) as the fraction words of an Fx (the caller ensures it is < 1).
#[inline(always)]
fn window<const W: usize>(x: &[u64], lo: i64) -> Fx<W> {
    let mut r = [0u64; W];
    for (i, w) in r.iter_mut().take(W - 1).enumerate() {
        *w = bits64(x, lo + 64 * i as i64);
    }
    Fx(r)
}

fn bitlen(x: &[u64]) -> i64 {
    for i in (0..x.len()).rev() {
        if x[i] != 0 {
            return 64 * i as i64 + 64 - x[i].leading_zeros() as i64;
        }
    }
    0
}

#[inline(always)]
fn bit(x: &[u64], i: i64) -> bool {
    i >= 0 && (i as usize) / 64 < x.len() && (x[(i as usize) / 64] >> (i % 64)) & 1 == 1
}

/// Whether every bit of x in [lo, hi) is `one` (true for an empty range).
fn bits_all(x: &[u64], lo: i64, hi: i64, one: bool) -> bool {
    let mut i = lo.max(0);
    while i < hi {
        let w = (i / 64) as usize;
        let s = i % 64;
        let n = (64 - s).min(hi - i);
        let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << s };
        let v = if w < x.len() { x[w] } else { 0 };
        let ok = if one { v & mask == mask } else { v & mask == 0 };
        if !ok {
            return false;
        }
        i += n;
    }
    true
}

// ------------------------------------------------------------------ tables

/// Fraction words of the tables, and their size with the integer word.
const TP: usize = 10;
const TW: usize = TP + 1;
type Tw = [u64; TW];
/// The type the tables are computed in (one guard word).
type T = Fx<12>;

/// 1 / k for k < INV_N (Horner coefficients of the logarithm series).
const INV_N: usize = 72;
/// 1 / k! for k < FACT_N (Horner coefficients of cos and sin).
const FACT_N: usize = 72;

struct Tables {
    ln2: Tw,
    half_pi: Tw,
    /// 512 / (257 + i) and ln(512 / (257 + i)) for i < 256 (first reduction).
    recip: Vec<Tw>,
    l1: Vec<Tw>,
    /// ln(1 + j / 2^15) for j < 128 (second reduction).
    l2: Vec<Tw>,
    /// cos and sin of pi j / 1024 for j <= 256.
    cos: Vec<Tw>,
    sin: Vec<Tw>,
    inv: Vec<Tw>,
    inv_fact: Vec<Tw>,
}

fn store(v: &T) -> Tw {
    let mut t = [0u64; TW];
    t.copy_from_slice(&v.0[12 - TW..]);
    t
}

/// atanh(1 / m) = sum_k 1 / ((2k + 1) m^(2k + 1)) for 3 <= m < 2^32 (error a
/// few hundred ulps of T at most: far below the tables' last word).
fn atanh_inv(m: u64) -> T {
    let m2 = m * m;
    let mut pw = T::int(1).div_u64(m);
    let mut sum = T::ZERO;
    let mut k = 0u64;
    while !pw.is_zero() {
        sum = sum.add(&pw.div_u64(2 * k + 1));
        pw = pw.div_u64(m2);
        k += 1;
    }
    sum
}

/// atan(1 / m) for m >= 2.
fn atan_inv(m: u64) -> T {
    let m2 = m * m;
    let mut pw = T::int(1).div_u64(m);
    let (mut pos, mut neg) = (T::ZERO, T::ZERO);
    let mut k = 0u64;
    while !pw.is_zero() {
        let t = pw.div_u64(2 * k + 1);
        if k % 2 == 0 {
            pos = pos.add(&t);
        } else {
            neg = neg.add(&t);
        }
        pw = pw.div_u64(m2);
        k += 1;
    }
    pos.sub(&neg)
}

fn build_tables() -> Tables {
    let ln2 = atanh_inv(3).shl(1);
    // Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    let pi = atan_inv(5).shl(4).sub(&atan_inv(239).shl(2));
    let recip = (0..256u64).map(|i| store(&T::int(512).div_u64(257 + i))).collect();
    // ln(512 / (257 + i)) = sum_{m = 257 + i}^{511} ln((m + 1) / m), ln((m + 1) / m) = 2 atanh(1 / (2m + 1))
    let mut l1 = vec![[0u64; TW]; 256];
    let mut acc = T::ZERO;
    for i in (0..256u64).rev() {
        if i < 255 {
            acc = acc.add(&atanh_inv(2 * (257 + i) + 1).shl(1));
        }
        l1[i as usize] = store(&acc);
    }
    let mut l2 = Vec::with_capacity(128);
    let mut acc2 = T::ZERO;
    for j in 0..128u64 {
        l2.push(store(&acc2));
        acc2 = acc2.add(&atanh_inv(2 * ((1 << 15) + j) + 1).shl(1));
    }
    // cos and sin of a = pi / 1024 by their series, then of j a by rotations (the
    // error grows by a few ulps of T per step: 2^-693 after 256 steps)
    let a = pi.shr(10);
    let (mut cp, mut cn, mut sp, mut sn) = (T::int(1), T::ZERO, T::ZERO, T::ZERO);
    let mut term = T::int(1);
    let mut n = 1u64;
    loop {
        term = term.mul(&a).div_u64(n);
        if term.is_zero() {
            break;
        }
        match n % 4 {
            0 => cp = cp.add(&term),
            1 => sp = sp.add(&term),
            2 => cn = cn.add(&term),
            _ => sn = sn.add(&term),
        }
        n += 1;
    }
    let (ca, sa) = (cp.sub(&cn), sp.sub(&sn));
    let (mut c, mut s) = (T::int(1), T::ZERO);
    let (mut cos, mut sin) = (Vec::with_capacity(257), Vec::with_capacity(257));
    for _ in 0..=256 {
        cos.push(store(&c));
        sin.push(store(&s));
        let c2 = c.mul(&ca).sub(&s.mul(&sa));
        s = s.mul(&ca).add(&c.mul(&sa));
        c = c2;
    }
    let inv = (0..INV_N as u64).map(|k| store(&if k == 0 { T::ZERO } else { T::int(1).div_u64(k) })).collect();
    let mut inv_fact = Vec::with_capacity(FACT_N);
    let mut f = T::int(1);
    for k in 0..FACT_N as u64 {
        if k > 0 {
            f = f.div_u64(k);
        }
        inv_fact.push(store(&f));
    }
    Tables { ln2: store(&ln2), half_pi: store(&pi.shr(1)), recip, l1, l2, cos, sin, inv, inv_fact }
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(build_tables)
}

// ---------------------------------------------------------------- sampler

/// sig = mant 2^exp at precision `prec`, as the reference holds it.
#[derive(Clone, Debug)]
pub struct Params {
    prec: u32,
    nbytes: usize,
    mant: [u64; IN],
    mant_words: usize,
    exp: i64,
    /// sig < 2^sbits
    sbits: i64,
    /// Fraction words of the evaluation.
    n: usize,
    /// The decision margin is 2^margin.
    margin: i64,
}

impl Params {
    /// The parameters for getrandbits(prec) uniforms and sig = mant 2^exp (mant
    /// little-endian bytes). `widen` enlarges the margin by 2^widen (tests).
    pub fn new(prec: u32, mant: &[u8], exp: i64, widen: u32) -> Result<Params, &'static str> {
        if !(16..=MAX_PREC).contains(&prec) {
            return Err("precision out of the native range");
        }
        let mut m = [0u64; IN];
        let used = mant.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        if used == 0 {
            return Err("sigma must be positive");
        }
        if used > 8 * IN {
            return Err("mantissa of sigma too long");
        }
        for (i, &b) in mant[..used].iter().enumerate() {
            m[i / 8] |= (b as u64) << (8 * (i % 8));
        }
        let sbits = bitlen(&m) + exp;
        if sbits > MAX_SIGMA_BITS || exp.abs() > 1 << 20 {
            return Err("sigma out of the native range");
        }
        let n = (((sbits + 64) + 63).div_euclid(64)).max(1) as usize;
        let margin = (sbits + 16 - 64 * n as i64).max(-100) + 1 + widen as i64;
        Ok(Params {
            prec,
            nbytes: prec.div_ceil(8) as usize,
            mant: m,
            mant_words: used.div_ceil(8),
            exp,
            sbits,
            n,
            margin,
        })
    }

    /// Bytes of one getrandbits(prec) call; a sample reads two.
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    /// Fraction words of the evaluation.
    pub fn words(&self) -> usize {
        self.n
    }

    pub fn sigma_bits(&self) -> i64 {
        self.sbits
    }

    /// Words of a sample's magnitude: |X| < 33 sig < 2^(s + 6), and rounding up adds one.
    pub fn mag_words(&self) -> usize {
        ((self.sbits + 7).max(1) as usize).div_ceil(64).min(MAG)
    }
}

/// getrandbits(prec) from its ceil(prec / 8) big-endian bytes.
fn read_bits(b: &[u8], prec: u32) -> [u64; IN] {
    let mut x = [0u64; IN + 1];
    for (i, &v) in b.iter().rev().enumerate() {
        x[i / 8] |= (v as u64) << (8 * (i % 8));
    }
    let d = (8 * b.len() as u32 - prec) as i64; // < 8
    let mut r = [0u64; IN];
    for (i, w) in r.iter_mut().enumerate() {
        *w = bits64(&x, d + 64 * i as i64);
    }
    r
}

/// sqrt(a) for 2^-8 <= a < 2^11 with an error below 32 ulps, checked exactly;
/// None if the check fails (never expected).
#[inline(always)]
fn sqrt_fx<const W: usize>(a: &Fx<W>) -> Option<Fx<W>> {
    let n = W - 1;
    let one = Fx::<W>::int(1);
    // y -> 1 / sqrt(a) by Newton (quadratic from the double's 2^-52)
    let mut y = Fx::<W>::from_f64(1.0 / a.to_f64().sqrt());
    let mut good = 50usize;
    while good < 64 * n + 16 {
        let t = a.mul(&y.mul(&y));
        y = if t.lt(&one) { y.add(&y.mul(&one.sub(&t)).shr(1)) } else { y.sub(&y.mul(&t.sub(&one)).shr(1)) };
        good *= 2;
    }
    // s = a y, then s + y (a - s^2) / 2, which removes the error a y carries
    let mut s = a.mul(&y);
    let s2 = s.mul(&s);
    s = if s2.lt(a) { s.add(&y.mul(&a.sub(&s2)).shr(1)) } else { s.sub(&y.mul(&s2.sub(a)).shr(1)) };
    // |sqrt(A 2^64n) - S| = |A 2^64n - S^2| / (sqrt(A 2^64n) + S) <= |rho| / S for the
    // integers A, S of a, s: accept when |rho| < 32 S
    let mut lhs = [0u64; PROD];
    lhs[n..n + W].copy_from_slice(&a.0);
    let mut sq = [0u64; PROD];
    for i in 0..W {
        let mut c = 0u128;
        for j in 0..W {
            let t = s.0[i] as u128 * s.0[j] as u128 + sq[i + j] as u128 + c;
            sq[i + j] = t as u64;
            c = t >> 64;
        }
        sq[i + W] = c as u64;
    }
    let len = 2 * W;
    let (big, small) = if lhs[..len].iter().rev().cmp(sq[..len].iter().rev()) == core::cmp::Ordering::Less {
        (&sq, &lhs)
    } else {
        (&lhs, &sq)
    };
    let mut rho = [0u64; PROD];
    let mut b = 0u64;
    for i in 0..len {
        let (d1, b1) = big[i].overflowing_sub(small[i]);
        let (d2, b2) = d1.overflowing_sub(b);
        rho[i] = d2;
        b = (b1 | b2) as u64;
    }
    (bitlen(&rho[..len]) <= bitlen(&s.0) - 1 + 5).then_some(s)
}

/// -ln u1 as (L, e) with -ln u1 = L 2^-2e: L >= 2^-8 when e = 0, L in [1/2, 2.02)
/// otherwise. None when u1 = 1 (the sample is 0). Error of L: below 10 ulps
/// (e = 0) or 12 ulps (e > 0).
#[inline(always)]
fn neg_log<const W: usize>(t: &Tables, b1: &[u64; IN], prec: u32) -> Option<(Fx<W>, u32)> {
    let n = (W - 1) as i64;
    let prec = prec as i64;
    // d = 2^p - 1 - b1: 1 - u1 = d / 2^p
    let mut d = [0u64; IN];
    for (i, w) in d.iter_mut().enumerate() {
        let lo = 64 * i as i64;
        let mask = if prec >= lo + 64 { u64::MAX } else if prec > lo { (1u64 << (prec - lo)) - 1 } else { 0 };
        *w = !b1[i] & mask;
    }
    let hd = bitlen(&d);
    if hd == 0 {
        return None;
    }
    if hd <= prec - 8 {
        // u1 > 1 - 2^-8: -ln(1 - t) = t g(t), g(t) = sum_k t^k / (k + 1), for
        // t = d / 2^p = m 2^-j, m in [1/2, 1), j >= 8 (relative precision kept)
        let tf = window::<W>(&d, prec - 64 * n); // t < 2^-8, error < 1 ulp
        let terms = ((64 * n + 3 + 7) / 8) as usize; // tail < 2^-(64n + 2)
        let mut g = Fx::<W>::tab(&t.inv[terms + 1]);
        for k in (1..=terms).rev() {
            g = Fx::<W>::tab(&t.inv[k]).add(&tf.mul(&g)); // Horner; error <= 3.1 ulps in all
        }
        let m = window::<W>(&d, hd - 64 * n); // error < 1 ulp
        let j = (prec - hd) as u32;
        let l = m.mul(&g); // error <= m dg + g dm + 1 < 5.6 ulps
        return Some(if j % 2 == 0 { (l, j / 2) } else { (l.shl(1), j.div_ceil(2)) });
    }
    // u1 <= 1 - 2^-8: u1 = x 2^-k with x = B / 2^h in [1/2, 1), B = b1 + 1
    let mut bb = *b1;
    for w in bb.iter_mut() {
        let (v, c) = w.overflowing_add(1);
        *w = v;
        if !c {
            break;
        }
    }
    let h = bitlen(&bb);
    let k = (prec - h) as u64;
    let x = window::<W>(&bb, h - 64 * n); // error < 1 ulp
    // -ln x = ln d_i - ln(x d_i), d_i = 512 / (257 + i) for x in [(256 + i) / 512, (257 + i) / 512):
    // x d_i in [1 - 1/257, 1)
    let i = (x.0[W - 2] >> 55) as usize - 256;
    let y1 = x.mul(&Fx::<W>::tab(&t.recip[i])); // <= x d_i; error < 4 ulps
    // -ln y1 = ln(1 + j 2^-15) - ln(y1 (1 + j 2^-15)) for j = floor((1 - y1) 2^15) < 128
    let one = Fx::<W>::int(1);
    let t1 = one.sub(&y1);
    let j = (t1.0[W - 2] >> 49) as usize;
    let y2 = y1.add(&y1.mul_u64(j as u64).shr(15)); // <= 1; error < 5.1 ulps
    let t2 = one.sub(&y2); // < 2^-14.4
    // -ln(1 - t2) = sum_{k >= 1} t2^k / k: tail below 2^-(64n + 2)
    let terms = ((64 * n + 2 + 13) / 14) as usize;
    let mut p = Fx::<W>::tab(&t.inv[terms]);
    for kk in (1..terms).rev() {
        p = Fx::<W>::tab(&t.inv[kk]).add(&t2.mul(&p));
    }
    let lt2 = t2.mul(&p); // error < 5.2 ulps (with t2's)
    // k ln 2 from the tables' 640 bits: error < 1 ulp + k 2^-640
    let kln2 = {
        let mut v = T::ZERO;
        v.0[12 - TW..].copy_from_slice(&t.ln2);
        let v = v.mul_u64(k);
        let mut a = [0u64; W];
        a.copy_from_slice(&v.0[12 - W..]);
        Fx(a)
    };
    let l = kln2.add(&Fx::<W>::tab(&t.l1[i])).add(&Fx::<W>::tab(&t.l2[j])).add(&lt2); // error < 10 ulps
    Some((l, 0))
}

/// |cos(2 pi u2)| and its sign; error below 10 ulps.
#[inline(always)]
fn cos_2pi<const W: usize>(t: &Tables, b2: &[u64; IN], prec: u32) -> (Fx<W>, bool) {
    let n = (W - 1) as i64;
    let prec = prec as i64;
    // 4 u2 = q + f: quadrant q, f in [0, 1)
    let q = (bit(b2, prec - 1) as u8) << 1 | bit(b2, prec - 2) as u8;
    let mut f = *b2;
    for (i, w) in f.iter_mut().enumerate() {
        let lo = 64 * i as i64;
        let keep = prec - 2;
        *w &= if keep >= lo + 64 { u64::MAX } else if keep > lo { (1u64 << (keep - lo)) - 1 } else { 0 };
    }
    let ff = window::<W>(&f, prec - 2 - 64 * n); // f, error < 1 ulp
    let fold = ff.0[W - 2] >> 63 == 1; // f >= 1/2
    // cos(pi/2 (q + f)) = +-cos or +-sin of phi = pi g / 2, g = f or 1 - f in [0, 1/2]
    let g = if fold { Fx::<W>::int(1).sub(&ff) } else { ff };
    let use_sin = (q & 1 == 1) ^ fold;
    let neg = q == 1 || q == 2;
    // phi = pi j / 1024 + psi, psi = pi h / 2 < pi / 1024, h = g - j / 512
    let j = (g.0[W - 2] >> 55) as usize; // <= 256
    let mut h = g;
    h.0[W - 2] &= (1u64 << 55) - 1;
    let psi = Fx::<W>::tab(&t.half_pi).mul(&h); // error < 3 ulps
    let psi2 = psi.mul(&psi);
    // cos psi = sum (-1)^k psi^2k / (2k)!, sin psi = psi sum (-1)^k psi^2k / (2k + 1)!:
    // alternating, tail below psi^(2 terms + 2) / (2 terms + 2)! < 2^-(64n + 2)
    let terms = ((64 * n + 2 + 15) / 16) as usize;
    let mut cp = Fx::<W>::tab(&t.inv_fact[2 * terms]);
    let mut sp = Fx::<W>::tab(&t.inv_fact[2 * terms + 1]);
    for k in (0..terms).rev() {
        cp = Fx::<W>::tab(&t.inv_fact[2 * k]).sub(&psi2.mul(&cp));
        sp = Fx::<W>::tab(&t.inv_fact[2 * k + 1]).sub(&psi2.mul(&sp));
    }
    let sp = psi.mul(&sp); // cos psi error < 3 ulps, sin psi < 4.1
    let (cj, sj) = (Fx::<W>::tab(&t.cos[j]), Fx::<W>::tab(&t.sin[j]));
    let c = if use_sin {
        sj.mul(&cp).add(&cj.mul(&sp)) // sin(phi_j + psi)
    } else {
        cj.mul(&cp).sub(&sj.mul(&sp)) // cos(phi_j + psi), >= 0 for phi <= pi / 4 + pi / 1024
    };
    (c, neg)
}

/// One sample: Ok((negative, magnitude)) or Err(()) when undecided.
#[inline(always)]
fn sample_w<const W: usize>(t: &Tables, p: &Params, b1: &[u64; IN], b2: &[u64; IN], mag: &mut [u64]) -> Result<bool, ()> {
    mag.fill(0);
    let Some((l, e)) = neg_log::<W>(t, b1, p.prec) else {
        return Ok(false); // u1 = 1: z = 0
    };
    let r = sqrt_fx(&l.shl(1)).ok_or(())?; // sqrt(-2 ln u1) 2^e; error < 150 ulps
    let (c, neg) = cos_2pi::<W>(t, b2, p.prec);
    let z = r.mul(&c); // |z| 2^e; error < 33 * 10 + 150 + 1 < 2^10 ulps
    approx_round(p, &z.0, e as i64, mag).map(|()| neg)
}

/// |X| = Z sig 2^-(64n + e) for the integer Z of z: the nearest integer, into
/// `mag`, if |X| is farther than 2^margin from every half-integer.
fn approx_round(p: &Params, z: &[u64], e: i64, mag: &mut [u64]) -> Result<(), ()> {
    let mut prod = [0u64; IN + MAX_FRAC + 2];
    for (i, &zi) in z.iter().enumerate() {
        let mut c = 0u128;
        for j in 0..p.mant_words {
            let v = zi as u128 * p.mant[j] as u128 + prod[i + j] as u128 + c;
            prod[i + j] = v as u64;
            c = v >> 64;
        }
        prod[i + p.mant_words] = c as u64;
    }
    // |X| = prod 2^-sh; sh >= bits(mant) + 64 > 0
    let sh = 64 * p.n as i64 + e - p.exp;
    let mb = p.margin + sh; // the margin in units of 2^-sh (> 0)
    let up = bit(&prod, sh - 1);
    // fraction below 1/2: decided unless the bits [mb, sh - 1) are all 1 (within 2^mb of 1/2);
    // at or above 1/2: decided if a bit in [mb + 1, sh - 1) is set (more than 2^mb above 1/2)
    let undecided = if up { bits_all(&prod, mb + 1, sh - 1, false) } else { bits_all(&prod, mb, sh - 1, true) };
    if undecided || mb + 2 >= sh || bitlen(&prod) - sh >= 64 * mag.len() as i64 {
        return Err(());
    }
    for (i, w) in mag.iter_mut().enumerate() {
        *w = bits64(&prod, sh + 64 * i as i64);
    }
    if up {
        for w in mag.iter_mut() {
            let (v, c) = w.overflowing_add(1);
            *w = v;
            if !c {
                break;
            }
        }
    }
    prod.fill(0);
    Ok(())
}

macro_rules! with_words {
    ($n:expr, $w:ident => $body:expr) => {
        match $n {
            1 => { const $w: usize = 2; $body }
            2 => { const $w: usize = 3; $body }
            3 => { const $w: usize = 4; $body }
            4 => { const $w: usize = 5; $body }
            5 => { const $w: usize = 6; $body }
            6 => { const $w: usize = 7; $body }
            7 => { const $w: usize = 8; $body }
            _ => { const $w: usize = 9; $body }
        }
    };
}

/// Samples of discrete_gaussian_bigint for `flags.len()` pairs of uniforms in
/// `bytes` (b1 then b2 per sample, `p.nbytes()` bytes each): with m =
/// `p.mag_words()`, sample i is (-1)^flags[i] mags[m i .. m (i + 1)]
/// (little-endian words), or flags[i] = UNDECIDED (compute it from its bytes
/// with the reference formula).
pub fn fill(p: &Params, bytes: &[u8], mags: &mut [u64], flags: &mut [u8]) {
    let count = flags.len();
    let m = p.mag_words();
    assert!(bytes.len() >= 2 * p.nbytes * count && mags.len() >= m * count);
    let t = tables();
    with_words!(p.n, W => {
        for i in 0..count {
            let at = 2 * p.nbytes * i;
            let mut b1 = read_bits(&bytes[at..at + p.nbytes], p.prec);
            let mut b2 = read_bits(&bytes[at + p.nbytes..at + 2 * p.nbytes], p.prec);
            let mag = &mut mags[m * i..m * (i + 1)];
            flags[i] = match sample_w::<W>(t, p, &b1, &b2, mag) {
                Ok(neg) => if neg && mag.iter().any(|&w| w != 0) { NEGATIVE } else { POSITIVE },
                Err(()) => UNDECIDED,
            };
            b1.fill(0);
            b2.fill(0);
        }
    })
}

/// The approximation of X for one pair of uniforms, for the tests:
/// Some((prod, sh, negative)) with |X| ~ prod 2^-sh, None when u1 = 1 or the
/// square root check fails.
pub fn approx(p: &Params, b1: &[u8], b2: &[u8]) -> Option<(Vec<u64>, i64, bool)> {
    let t = tables();
    let (b1, b2) = (read_bits(b1, p.prec), read_bits(b2, p.prec));
    with_words!(p.n, W => {
        let (l, e) = neg_log::<W>(t, &b1, p.prec)?;
        let r = sqrt_fx(&l.shl(1))?;
        let (c, neg) = cos_2pi::<W>(t, &b2, p.prec);
        let z = r.mul(&c);
        let mut prod = vec![0u64; IN + MAX_FRAC + 2];
        for (i, &zi) in z.0.iter().enumerate() {
            let mut cy = 0u128;
            for j in 0..p.mant_words {
                let v = zi as u128 * p.mant[j] as u128 + prod[i + j] as u128 + cy;
                prod[i + j] = v as u64;
                cy = v >> 64;
            }
            prod[i + p.mant_words] = cy as u64;
        }
        Some((prod, 64 * p.n as i64 + e as i64 - p.exp, neg))
    })
}

/// getrandbits of sampling_utils.XofRandom: call `counter` of the seed's
/// SHAKE-256 stream, `out.len()` bytes.
pub fn xof_call(head: &[u8], counter: u64, out: &mut [u8]) {
    let mut msg = Vec::with_capacity(head.len() + 8);
    msg.extend_from_slice(head);
    msg.extend_from_slice(&counter.to_be_bytes());
    crate::keccak::shake256(&msg, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f64(v: &Tw) -> f64 {
        Fx::<3>::tab(v).to_f64()
    }

    #[test]
    fn tables_are_consistent() {
        let t = tables();
        assert!((to_f64(&t.ln2) - std::f64::consts::LN_2).abs() < 1e-15);
        assert!((to_f64(&t.half_pi) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        for i in 0..256 {
            let d = 512.0 / (257.0 + i as f64);
            assert!((to_f64(&t.l1[i]) - d.ln()).abs() < 1e-15, "l1[{i}]");
            assert!((to_f64(&t.recip[i]) - d).abs() < 1e-15);
        }
        for j in 0..128 {
            assert!((to_f64(&t.l2[j]) - (j as f64 / 32768.0).ln_1p()).abs() < 1e-17);
        }
        for j in 0..=256 {
            let a = std::f64::consts::PI * j as f64 / 1024.0;
            assert!((to_f64(&t.cos[j]) - a.cos()).abs() < 1e-15 && (to_f64(&t.sin[j]) - a.sin()).abs() < 1e-15);
            // cos^2 + sin^2 = 1 to the tables' precision
            let (c, s) = (Fx::<11>::tab(&t.cos[j]), Fx::<11>::tab(&t.sin[j]));
            let one = c.mul(&c).add(&s.mul(&s));
            let d = if one.lt(&Fx::int(1)) { Fx::int(1).sub(&one) } else { one.sub(&Fx::int(1)) };
            assert!(bitlen(&d.0) < 8, "j = {j}");
        }
    }

    #[test]
    fn small_widths_match_doubles() {
        // sigma = 1000 (exact), p = 138: compare with f64 Box-Muller
        let p = Params::new(138, &1000u64.to_le_bytes(), 0, 0).unwrap();
        let nb = p.nbytes();
        let mut seed = 0x9e3779b97f4a7c15u64;
        for _ in 0..2000 {
            let mut b = vec![0u8; 2 * nb];
            for v in b.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = (seed >> 56) as u8;
            }
            let x1 = read_bits(&b[..nb], 138);
            let x2 = read_bits(&b[nb..], 138);
            let u1 = (x1[2] as f64 * 2f64.powi(128) + x1[1] as f64 * 2f64.powi(64) + 1.0) / 2f64.powi(138);
            let u2 = (x2[2] as f64 * 2f64.powi(128) + x2[1] as f64 * 2f64.powi(64)) / 2f64.powi(138);
            let want = 1000.0 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let (prod, sh, neg) = approx(&p, &b[..nb], &b[nb..]).unwrap();
            let mut v = 0.0f64;
            for (i, &w) in prod.iter().enumerate().filter(|(_, &w)| w != 0) {
                v += w as f64 * 2f64.powi(64 * i as i32 - sh as i32);
            }
            let v = if neg { -v } else { v };
            assert!((v - want).abs() < 1e-9 * want.abs().max(1.0), "{v} vs {want}");
        }
    }

    #[test]
    fn bit_ranges() {
        let x = [u64::MAX, 0b1011, 0];
        assert!(bits_all(&x, 0, 64, true) && bits_all(&x, 64, 66, true) && !bits_all(&x, 64, 67, true));
        assert!(bits_all(&x, 68, 200, false) && !bits_all(&x, 60, 70, false) && bits_all(&x, 5, 5, false));
    }
}
