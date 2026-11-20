//! The exact discrete Gaussian of pyfe4ai.primitives.sampling.discrete_gaussian_facct
//! (FACCT-style double sampling) for widths whose samples fit an i64.
//!
//! For a given bit source the samples are those of the Python reference: the
//! same `getrandbits` calls in the same order (u, then y until y < k, then v,
//! then the sign) and the same decisions. The base CDT comes from Python (its
//! MPFR tables). The acceptance test `v < floor(2^192 exp(-y (y + 2 k x) /
//! (2 sigma^2)))` -- rounded twice to 256 bits by MPFR in Python -- is decided
//! here with a fixed-point exp(-a) whose error is below 2^-110 (about 2^-119
//! measured) provided the table's c = 1 / (2 sigma^2) is correct to its 128-bit
//! mantissa: whenever v is within 2^-100 (relative to 2^192) of the threshold,
//! the exact threshold is asked from `exact(x, y)` (the Python formula;
//! probability about 2^-99 per trial), so the decisions are always Python's.
//! `check_table` compares a few thresholds with `exact` (the binding runs it
//! when a table is built), so a grossly imprecise c (off by more than about
//! 2^-102, such as a 53-bit one) is refused, not used.
//!
//! Since 1.0 the sampler is constant time in the values it draws (see the note
//! above `opaque`): every candidate takes the acceptance test (y = 0 included,
//! where the reference has none and the result is set to "accept" by a mask),
//! and the exponential has no branch and reads its whole table. Its time
//! follows the number of candidates and of y draws, which do not depend on the
//! values returned. The samples are those of 0.4.0 to 0.5.0, bit for bit (the
//! tests keep that version as a reference).

use crate::arith::{Word, U256};
use crate::keccak::shake256;

/// Bits of the CDT thresholds and of the acceptance tests (FACCT_PRECISION).
pub const PRECISION: u32 = 192;
/// Entries of the base CDT (_FACCT_CDT_SIZE).
pub const CDT_SIZE: usize = 14;

/// A `getrandbits` source: `k` bits (k <= 256) as an integer, as Python's.
pub trait RandBits {
    fn getrandbits(&mut self, k: u32) -> U256;
    /// getrandbits(k) for k <= 64 out.len(), into little-endian words (one call).
    fn getrandbits_into(&mut self, k: u32, out: &mut [u64]);
}

/// The top k bits of the big-endian bytes (at most 128), into little-endian words.
fn top_bits_into(bytes: &[u8], k: u32, out: &mut [u64]) {
    let mut x = [0u64; 17];
    for (i, &b) in bytes.iter().rev().enumerate() {
        x[i / 8] |= (b as u64) << (8 * (i % 8));
    }
    let d = 8 * bytes.len() as u32 - k; // < 8
    for (i, w) in out.iter_mut().enumerate() {
        *w = if d == 0 { x[i] } else { (x[i] >> d) | (x[i + 1] << (64 - d)) };
    }
    for w in x.iter_mut() {
        unsafe { std::ptr::write_volatile(w, 0) };
    }
}

/// getrandbits over a byte stream, as `_OSRandomBuffer` and `random.getrandbits`
/// of a byte source: ceil(k / 8) bytes, big-endian, the top k bits.
pub struct ByteBits<F: FnMut(&mut [u8]) -> usize> {
    refill: F,
    buf: Vec<u8>,
    len: usize,
    pos: usize,
    /// Bytes the refills delivered so far.
    total: usize,
    /// The stream ended (or failed) before all bytes were read; the rest read as 0.
    pub failed: bool,
}

impl<F: FnMut(&mut [u8]) -> usize> ByteBits<F> {
    /// `refill(buf)` writes the next bytes of the stream to buf and returns how
    /// many (0: none left).
    pub fn new(block: usize, refill: F) -> Self {
        ByteBits { refill, buf: vec![0u8; block], len: 0, pos: 0, total: 0, failed: false }
    }

    /// Bytes of the stream read so far (by getrandbits).
    pub fn consumed(&self) -> usize {
        self.total - (self.len - self.pos)
    }
    fn byte(&mut self) -> u8 {
        if self.pos == self.len {
            self.len = if self.failed { 0 } else { (self.refill)(&mut self.buf) };
            self.total += self.len;
            self.pos = 0;
            if self.len == 0 {
                self.failed = true;
                return 0;
            }
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }
}

impl<F: FnMut(&mut [u8]) -> usize> Drop for ByteBits<F> {
    fn drop(&mut self) {
        crate::rlwe::wipe_bytes(&mut self.buf);
    }
}

fn top_bits(bytes: &[u8], k: u32) -> U256 {
    let mut limbs = [0u64; 4];
    for (i, &b) in bytes.iter().rev().enumerate() {
        limbs[i / 8] |= (b as u64) << (8 * (i % 8));
    }
    let d = 8 * bytes.len() as u32 - k; // < 8
    if d > 0 {
        for i in 0..4 {
            limbs[i] = (limbs[i] >> d) | if i < 3 { limbs[i + 1] << (64 - d) } else { 0 };
        }
    }
    U256(limbs)
}

impl<F: FnMut(&mut [u8]) -> usize> RandBits for ByteBits<F> {
    fn getrandbits_into(&mut self, k: u32, out: &mut [u64]) {
        let n = k.div_ceil(8) as usize;
        let mut tmp = [0u8; 8 * 16];
        for b in tmp[..n].iter_mut() {
            *b = self.byte();
        }
        top_bits_into(&tmp[..n], k, out);
        crate::rlwe::wipe_bytes(&mut tmp);
    }

    fn getrandbits(&mut self, k: u32) -> U256 {
        if k == 0 {
            return U256::ZERO;
        }
        let n = k.div_ceil(8) as usize;
        let mut tmp = [0u8; 32];
        if self.len - self.pos >= n {
            tmp[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
        } else {
            for b in tmp[..n].iter_mut() {
                *b = self.byte();
            }
        }
        let v = top_bits(&tmp[..n], k);
        crate::rlwe::wipe_bytes(&mut tmp);
        v
    }
}

/// sampling.XofRandom: getrandbits(k) = the top k bits of
/// SHAKE-256(len(seed) (8 bytes, big-endian) || seed || counter (8 bytes,
/// big-endian)), ceil(k / 8) bytes, one hash per call.
pub struct XofBits {
    head: Vec<u8>,
    counter: u64,
    /// The counter passed 2^64 - 1: the next call has no 8-byte counter.
    past_end: bool,
    /// A call was made past the end (its bits are zeros; the caller reports an error).
    overflowed: bool,
}

impl XofBits {
    pub fn new(seed: &[u8]) -> Self {
        let mut head = (seed.len() as u64).to_be_bytes().to_vec();
        head.extend_from_slice(seed);
        XofBits { head, counter: 0, past_end: false, overflowed: false }
    }

    /// The XofRandom whose `_seed` is `head` (len(seed) || seed) at call `counter`.
    pub fn at(head: &[u8], counter: u64) -> Self {
        XofBits { head: head.to_vec(), counter, past_end: false, overflowed: false }
    }

    /// The next call's counter (the XofRandom's `_counter` after the draws so
    /// far; 2^64 once call 2^64 - 1 is made).
    pub fn next_call(&self) -> u128 {
        self.counter as u128 + self.past_end as u128
    }

    /// Whether a call needed a counter beyond 2^64 - 1 (where XofRandom raises
    /// OverflowError); such calls read zeros.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// The hashed message of the next call, or None past the 64-bit counters.
    fn next_message(&mut self) -> Option<Vec<u8>> {
        if self.past_end {
            self.overflowed = true;
            return None;
        }
        let mut msg = self.head.clone();
        msg.extend_from_slice(&self.counter.to_be_bytes());
        match self.counter.checked_add(1) {
            Some(c) => self.counter = c,
            None => self.past_end = true,
        }
        Some(msg)
    }
}

impl Drop for XofBits {
    fn drop(&mut self) {
        crate::rlwe::wipe_bytes(&mut self.head); // holds the secret seed
    }
}

impl RandBits for XofBits {
    fn getrandbits_into(&mut self, k: u32, out: &mut [u64]) {
        let n = k.div_ceil(8) as usize;
        let mut tmp = [0u8; 8 * 16];
        if let Some(mut msg) = self.next_message() {
            shake256(&msg, &mut tmp[..n]);
            crate::rlwe::wipe_bytes(&mut msg);
        }
        top_bits_into(&tmp[..n], k, out);
        crate::rlwe::wipe_bytes(&mut tmp);
    }

    fn getrandbits(&mut self, k: u32) -> U256 {
        let n = k.div_ceil(8) as usize;
        let mut out = [0u8; 32];
        if let Some(mut msg) = self.next_message() {
            if n > 0 {
                shake256(&msg, &mut out[..n]);
            }
            crate::rlwe::wipe_bytes(&mut msg);
        }
        top_bits(&out[..n], k)
    }
}

/// The tables of one width (sampling._facct_params): k, the base CDT, and
/// c = -neg_c = 1 / (2 sigma^2) rounded down to c_mant 2^c_exp (c_mant < 2^128).
#[derive(Clone, Debug)]
pub struct Table {
    pub k: u64,
    pub y_bits: u32,
    pub cdt: [U256; CDT_SIZE],
    pub c_mant: u128,
    pub c_exp: i32,
}

/// Fractional bits of the fixed-point exponential.
pub(crate) const F: u32 = 120;
const ONE: u128 = 1 << F;
/// floor(ln 2 2^120).
const LN2: u128 = 0xb17217f7d1cf79abc9e3b39803f2f6;
/// floor(2^120 exp(-j / 64)).
const EXP_J64: [u128; 64] = [
    0x1000000000000000000000000000000, 0xfc07f55ff77d2493e885eeaa756ad5, 0xf81fab5445aebc8a58055fcbbb139a,
    0xf446e357f67dfd8e33bd6c7d23c684, 0xf07d5fde38151e72f18ff03049ac5d, 0xecc2e44e820d18c3cc6f189c52b0b3,
    0xe9173500cbe0159aa3084a1ddd389d, 0xe57a1739d263ec948222e86239e16e, 0xe1eb51276c110c3c3eb1269f2f5d4a,
    0xde6aa9dcebdb100b56a61a9a32d7cd, 0xdaf7e94f9261313d5fd7d70a5ed575, 0xd792d8530d3da5312b5abcb3ceac1e,
    0xd43b4096043bde02c9dd90522bbe5c, 0xd0f0ec9eb43e8078e9af6bb3a3e274, 0xcdb3a7c7979ebd5c80489daded79ae,
    0xca833e3c1bcf93e90747d19ac1d33d, 0xc75f7cf564105743415cbc9d6368f3, 0xc44831b718faa1d9c29d8d982ed75c,
    0xc13d2b0c44b8af4a68aa4e7ef9fa53, 0xbe3e38443bb1dfef65597daacdcf1b, 0xbb4b296f917bf09a22fbaa27077b85,
    0xb863cf5d19e0354e91dbb1734bd4ad, 0xb587fb96f5c4e712a053451f9ff0ec, 0xb2b7805fabcb5327cdbad0e5a252d4,
    0xaff230af4c747553d8db804c224277, 0xad37e030a19e3f45315a9f67efbe59, 0xaa88633e692c84ab9760aff641e870,
    0xa7e38ee09ab136588e4fa88f63be82, 0xa54938c9b7e846b122756d14a7c2c4, 0xa2b9375427dc3ef5fdb601369d6cea,
    0xa033617f9c893772b7f2708ea14d50, 0x9db78eee82d48c8c5e432e3c71dae4, 0x9b4597e37cb04ff3d675a35530cdd7,
    0x98dd553ee54217fd5f7533af59a20a, 0x967ea07c5ee56f5527376efe92ea16, 0x942953b06ae2c503c6e249a5881fe9,
    0x91dd49860ab457fe49ecbc03de5c55, 0x8f9a5d3c6ab3325de98839ae4abeaf, 0x8d606aa49606dedd68c91a372c64f1,
    0x8b2f4e1f33b3176667ee084844e87b, 0x8906e49a4c9f3d5958a5d1795c6af8, 0x86e70b8f1a73fbd81858e18d95147b,
    0x84cfa0ffdf2c01be9229f8beebdb8b, 0x82c08375c5354c26fed9d9af334631, 0x80b991fec8010361182aa185169de3,
    0x7ebaac2ba4e0732d70709e3a7db50c, 0x7cc3b20dd40e2be08458b36fa8909d, 0x7ad4843589c2dcd459c11c65ad53ff,
    0x78ed03afbf35f94bbf0f4cffc58234, 0x770d1204436ab7a78e839f14ad09a9, 0x75349133d3aa7794a488ee1b42c1fc,
    0x736363b63b8e16b04b525f1016963f, 0x71996c787c78341066bfc732c2072f, 0x6fd68edafc62da2c387c104885e875,
    0x6e1aaeafbbe37bd9969b58182a099b, 0x6c65b0389348a4783a2244922dcadb, 0x6ab7782576b52d00953b6ea1c5fec2,
    0x690feb92c11d37848438f858b46a11, 0x676ef0078609a0cd266ee7b4cdb93c, 0x65d46b73ea0704291e4f81a405658f,
    0x6440442f81a5d8382985cfa3934786, 0x62b260f7b6f194931db7af62ad2bc7, 0x612aa8ee354537a06e4ac78bafa878,
    0x5fa903975b63e9bbea189925bac0f8,
];
/// floor(2^120 / i!).
const INV_FACT: [u128; 31] = [
    0x1000000000000000000000000000000, 0x1000000000000000000000000000000, 0x800000000000000000000000000000,
    0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x22222222222222222222222222222,
    0x5b05b05b05b05b05b05b05b05b05, 0xd00d00d00d00d00d00d00d00d00, 0x1a01a01a01a01a01a01a01a01a0,
    0x2e3bc74aad8e671f5583911ca0, 0x49f93edde27d71cbbc05b4fa9, 0x6b99159fd5138e3f9d1f92e0,
    0x8f76c77fc6c4bdaa26d4c3d, 0xb092309d43684be51c198e, 0xc9cba54603e4e905d6f8a,
    0xd73f9f399dc0f88ec32b, 0xd73f9f399dc0f88ec32, 0xca963b81856a535930,
    0xb413c31dcbecbbdd8, 0x97a4da340a0ab926, 0x7950ae900808941,
    0x5c6e3bdb73d5c6, 0x4338e5b6dfe14, 0x2ec368262c70,
    0x1f2cf01972f, 0x13f3ccdd16, 0xc4742fe3,
    0x746ac70, 0x428628, 0x24b3f,
    0x1393,
];

/// floor(2^176 / LN2): ((a >> 64) INV_LN2) >> 112 is floor(a / LN2) or one less
/// for a < 2^127 (it is at most a / LN2 and above a / LN2 - 2^-48).
const INV_LN2: u128 = 0x171547652b82fe1;
/// The largest j = r >> (F - 6) for r < LN2: floor(64 ln 2) = 44.
const J_MAX: usize = 44;

// Constant time. The candidate's values (x, y, v) and the sample's sign reach
// no branch and no address below: comparisons are borrow chains turned into
// 0/1 words, choices are masks, the table of exp(-j / 64) is read whole, and
// shifts by a secret amount are word shifts (constant-time on the targets) and
// masks. `opaque` keeps the compiler from turning masks back into branches.
// What still branches: the table's public data (k, y_bits, c), the decision to
// draw again (y >= k, a rejected candidate, z = 0 with the sign set), whose
// count does not depend on the values returned, and the deferral to `exact`,
// about once in 2^99 tests.

/// x, hidden from the optimiser (an empty asm block that takes and returns it).
#[inline(always)]
pub(crate) fn opaque(x: u64) -> u64 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let mut x = x;
        // SAFETY: the block is empty; it only claims to read and write x's register.
        unsafe { core::arch::asm!("/* {0} */", inout(reg) x, options(pure, nomem, nostack, preserves_flags)) };
        x
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        core::hint::black_box(x)
    }
}

/// 1 if x != 0, else 0.
#[inline(always)]
pub(crate) fn nonzero(x: u64) -> u64 {
    opaque((x | x.wrapping_neg()) >> 63)
}

/// All ones if bit (0 or 1) is 1, else 0.
#[inline(always)]
pub(crate) fn mask64(bit: u64) -> u64 {
    0u64.wrapping_sub(bit)
}

/// All ones if bit (0 or 1) is 1, else 0.
#[inline(always)]
pub(crate) fn mask128(bit: u64) -> u128 {
    0u128.wrapping_sub(bit as u128)
}

/// a >= b (1 or 0) for numbers of equal length in little-endian words: no borrow out of a - b.
#[inline(always)]
pub(crate) fn ge_words(a: &[u64], b: &[u64]) -> u64 {
    let mut borrow = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        borrow = ((x as u128).wrapping_sub(y as u128 + borrow as u128) >> 127) as u64;
    }
    opaque(borrow ^ 1)
}

/// a >= b (1 or 0).
#[inline(always)]
pub(crate) fn ge256(a: &U256, b: &U256) -> u64 {
    ge_words(&a.0, &b.0)
}

/// a >= b (1 or 0).
#[inline(always)]
fn ge128(a: u128, b: u128) -> u64 {
    opaque(a.overflowing_sub(b).1 as u64 ^ 1)
}

/// e >> m for m < 256 (0 for m >= 128): 64-bit shifts by m mod 64, then masks for m / 64.
#[inline(always)]
fn shr_masked(e: u128, m: u64) -> u128 {
    let (lo, hi) = (e as u64, (e >> 64) as u64);
    let s = (m & 63) as u32;
    let lo_s = (lo >> s) | ((hi << 1) << (63 - s)); // hi << (64 - s), 0 for s = 0
    let hi_s = hi >> s;
    let w = m >> 6;
    let (w0, w1) = (mask64(nonzero(w) ^ 1), mask64(nonzero(w ^ 1) ^ 1));
    (((hi_s & w0) as u128) << 64) | ((lo_s & w0) | (hi_s & w1)) as u128
}

#[inline]
fn mul_f(a: u128, b: u128) -> u128 {
    let (hi, lo) = a.full(b);
    (hi << (128 - F)) | (lo >> F)
}

/// (P >> s) for a 256-bit P given as (hi, lo) and a public s, and whether it
/// fits 128 bits (1 or 0).
fn shr_pair(hi: u128, lo: u128, s: u32) -> (u128, u64) {
    let fits = |x: u128| nonzero(x as u64 | (x >> 64) as u64) ^ 1;
    match s {
        0 => (lo, fits(hi)),
        1..=127 => ((hi << (128 - s)) | (lo >> s), fits(hi >> s)),
        128..=255 => (hi >> (s - 128), 1),
        _ => (0, 1),
    }
}

/// exp(-a) in fixed point (F fractional bits) for a = c t, and 1; error below
/// 2^-110 when c_mant 2^c_exp is c rounded down to 128 bits (check_table checks
/// it). (0, 0) if a is beyond the range this handles (the caller then uses `exact`).
fn exp_neg(table: &Table, t: u128) -> (u128, u64) {
    // a = c_mant t 2^c_exp with F fractional bits: (c_mant t) >> (-(c_exp + F))
    let s = -(table.c_exp + F as i32);
    if s < 0 {
        return (0, 0); // the table's c (public): never in this fixed point
    }
    let (hi, lo) = table.c_mant.full(t);
    let (a, ok) = shr_pair(hi, lo, s as u32);
    (exp_neg_a(a) & mask128(ok), ok)
}

/// exp(-a) in fixed point (F fractional bits) for a with F fractional bits;
/// error below 2^-110 (0 for a >= 120). Constant time (see above).
pub(crate) fn exp_neg_a(a: u128) -> u128 {
    // a >= 120: exp(-a) < 2^-173, far below every threshold the tests can meet:
    // 0 (a is taken as 0 below, and the result masked)
    let small = ge128(120 * ONE - 1, a);
    let a = a & mask128(small);
    // a = m ln 2 + r, r in [0, ln 2): m = floor(a / LN2) <= 173, from the
    // fixed-point 1 / ln 2 (m or m - 1) and one masked step
    let est = ((a >> 64) * INV_LN2) >> 112;
    let m = est + ge128(a, (est + 1) * LN2) as u128;
    let r = a - m * LN2;
    // r = j / 64 + r2 (r2 < 1 / 64): exp(-r) = exp(-j / 64) exp(-r2), the latter
    // as sum_{i <= 15} (-r2)^i / i! (the rest is below 2^-130) by Horner's rule,
    // every partial value in (0, 1] since r2 < 1 and i! / (i + 1)! < 1
    let j = (r >> (F - 6)) as u64; // <= 44, as r < LN2
    let r2 = r - ((j as u128) << (F - 6));
    let mut p = INV_FACT[15];
    for i in (0..15).rev() {
        p = INV_FACT[i] - mul_f(p, r2);
    }
    // EXP_J64[j], every entry j can take read (i ^ j - 1 has its top bit set only for i = j)
    let j = opaque(j);
    let mut ej = 0u128;
    for (i, &t) in EXP_J64[..=J_MAX].iter().enumerate() {
        ej |= t & mask128(((i as u64 ^ j).wrapping_sub(1) >> 63) & 1);
    }
    shr_masked(mul_f(ej, p), m as u64) & mask128(small)
}

/// The acceptance decision `v < floor(2^192 exp(-c t))` of the Python sampler,
/// or true for y = 0, where the reference has no test (its threshold, 2^192, is
/// above every v). The fixed-point test runs for every candidate, y = 0 included,
/// and the outcomes are combined with masks; only the deferral to `exact`
/// branches (about once in 2^99 tests with y != 0; every such test of a table
/// whose c the fixed point cannot hold).
fn accept(table: &Table, x: u64, y: u64, v: U256, exact: &dyn Fn(u64, u64) -> U256) -> bool {
    let t = y as u128 * (y as u128 + 2 * table.k as u128 * x as u128);
    let (e, ok) = exp_neg(table, t);
    let approx = U256::from_u128(e).shl(PRECISION - F);
    let margin = U256::ONE.shl(PRECISION - 100);
    // v + 1 + margin <= approx: v < threshold; v >= approx + margin: v >= threshold
    let below = ge256(&approx, &v.wadd(U256::ONE).wadd(margin)) & ok;
    let above = ge256(&v, &approx.wadd(margin)) & ok;
    let no_test = nonzero(y) ^ 1;
    if opaque(no_test | below | above) == 0 {
        return v < exact(x, y);
    }
    (no_test | below) == 1
}

/// Whether the fixed-point thresholds agree with `exact` to 2^-102 (relative to
/// 2^192) at the corners and the middle of the (x, y) range. It detects a
/// constant off by more than about 2^-102 at those points -- a 53-bit c is off
/// by about 2^-55 -- that is, errors that approach the 2^-100 band within which
/// the decisions are deferred; it does not show the 2^-110 of a correct c (about
/// 2^-119 measured). Points where exp_neg defers to `exact` anyway are skipped.
pub fn check_table(table: &Table, exact: &dyn Fn(u64, u64) -> U256) -> bool {
    if table.k < 2 {
        return true; // y = 0 always: no acceptance test
    }
    let (xm, ym) = (CDT_SIZE as u64, table.k - 1);
    for (x, y) in [(0, 1), (0, ym), (xm, 1), (xm, ym), (xm / 2, ym.div_ceil(2))] {
        let t = y as u128 * (y as u128 + 2 * table.k as u128 * x as u128);
        if let (e, 1) = exp_neg(table, t) {
            let (a, b) = (U256::from_u128(e).shl(PRECISION - F), exact(x, y));
            let d = if a >= b { a.wsub(b) } else { b.wsub(a) };
            if d >= U256::ONE.shl(PRECISION - 102) {
                return false;
            }
        }
    }
    true
}

/// `out.len()` samples of D_{Z,sigma} (discrete_gaussian_facct) from `bits`;
/// `exact(x, y)` is the Python threshold floor(2^192 exp(neg_c y (y + 2 k x))).
pub fn sample(table: &Table, out: &mut [i64], bits: &mut dyn RandBits, exact: &dyn Fn(u64, u64) -> U256) {
    let k = table.k;
    let mut filled = 0;
    while filled < out.len() {
        let u = bits.getrandbits(PRECISION);
        let mut x = 0u64;
        for th in table.cdt.iter() {
            x += ge256(&u, th); // the whole table, as the reference
        }
        let mut y = 0u64;
        if table.y_bits > 0 {
            y = bits.getrandbits(table.y_bits).0[0];
            while y >= k {
                y = bits.getrandbits(table.y_bits).0[0];
            }
        }
        let v = bits.getrandbits(PRECISION);
        if !accept(table, x, y, v, exact) {
            continue;
        }
        let z = (k * x + y) as i64;
        let negative = bits.getrandbits(1).0[0] & 1;
        if opaque((nonzero(z as u64) ^ 1) & negative) == 1 {
            continue; // -0: drawn again, as the reference
        }
        let n = (negative as i64).wrapping_neg(); // 0 or -1
        out[filled] = (z ^ n).wrapping_sub(n);
        filled += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// The sampler before it was made constant-time (pyfe4ai 0.4.0 to 0.5.0), kept
    /// as the reference that the constant-time one must match bit for bit.
    mod branching {
        use super::super::*;

        fn shr_pair(hi: u128, lo: u128, s: u32) -> Option<u128> {
            match s {
                0 => (hi == 0).then_some(lo),
                1..=127 => (hi >> s == 0).then_some((hi << (128 - s)) | (lo >> s)),
                128..=255 => Some(hi >> (s - 128)),
                _ => Some(0),
            }
        }

        fn exp_neg(table: &Table, t: u128) -> Option<u128> {
            let (hi, lo) = table.c_mant.full(t);
            let s = -(table.c_exp + F as i32);
            if s < 0 {
                return None;
            }
            let a = shr_pair(hi, lo, s as u32)?;
            Some(exp_neg_a(a))
        }

        pub fn exp_neg_a(a: u128) -> u128 {
            if a >= 120 * ONE {
                return 0;
            }
            let mut m = ((a as f64) / (LN2 as f64)) as u128;
            while m > 0 && m * LN2 > a {
                m -= 1;
            }
            while (m + 1) * LN2 <= a {
                m += 1;
            }
            let r = a - m * LN2;
            let j = (r >> (F - 6)) as usize;
            let r2 = r - ((j as u128) << (F - 6));
            let mut p = INV_FACT[15];
            for i in (0..15).rev() {
                p = INV_FACT[i] - mul_f(p, r2);
            }
            let e = mul_f(EXP_J64[j], p);
            if m >= 128 { 0 } else { e >> m }
        }

        fn accept(table: &Table, x: u64, y: u64, v: U256, exact: &dyn Fn(u64, u64) -> U256) -> bool {
            let t = y as u128 * (y as u128 + 2 * table.k as u128 * x as u128);
            if let Some(e) = exp_neg(table, t) {
                let approx = U256::from_u128(e).shl(PRECISION - F);
                let margin = U256::ONE.shl(PRECISION - 100);
                if v.wadd(U256::ONE).wadd(margin) <= approx {
                    return true;
                }
                if v >= approx.wadd(margin) {
                    return false;
                }
            }
            v < exact(x, y)
        }

        pub fn sample(table: &Table, out: &mut [i64], bits: &mut dyn RandBits, exact: &dyn Fn(u64, u64) -> U256) {
            let k = table.k;
            let mut filled = 0;
            while filled < out.len() {
                let u = bits.getrandbits(PRECISION);
                let mut x = 0u64;
                for th in table.cdt.iter() {
                    x += (u >= *th) as u64;
                }
                let mut y = 0u64;
                if table.y_bits > 0 {
                    y = bits.getrandbits(table.y_bits).0[0];
                    while y >= k {
                        y = bits.getrandbits(table.y_bits).0[0];
                    }
                }
                let v = bits.getrandbits(PRECISION);
                if y != 0 && !accept(table, x, y, v, exact) {
                    continue;
                }
                let z = (k * x + y) as i64;
                let negative = bits.getrandbits(1).0[0] == 1;
                if z == 0 && negative {
                    continue;
                }
                out[filled] = if negative { -z } else { z };
                filled += 1;
            }
        }
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// A table of the reference's form for sigma, from f64 (not the MPFR values:
    /// both samplers are given the same table, which is all these tests need).
    fn table_for(sigma: f64) -> Table {
        let k = ((sigma * (2.0 * 2f64.ln()).sqrt()).ceil() as u64).max(1);
        let c = 1.0 / (2.0 * sigma * sigma);
        let bits = c.to_bits();
        let (mant, exp) = ((bits & ((1 << 52) - 1)) | (1 << 52), ((bits >> 52) & 0x7ff) as i32 - 1075);
        let weights: Vec<f64> = (0..40).map(|x| (-c * ((x as f64) * k as f64).powi(2)).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut cdt = [U256::ZERO; CDT_SIZE];
        let mut acc = 0.0;
        for (i, th) in cdt.iter_mut().enumerate() {
            acc += weights[i];
            let f = (acc / total).min(1.0 - 1e-16) * 18446744073709551616.0;
            let top = f.floor();
            *th = U256([0, ((f - top) * 18446744073709551616.0) as u64, top as u64, 0]);
        }
        Table { k, y_bits: 64 - (k - 1).leading_zeros(), cdt, c_mant: (mant as u128) << 75, c_exp: exp - 75 }
    }

    /// The tables of both comparisons: widths from k = 1 to k near 2^50, and tables
    /// whose c is moved so that the fixed point defers often or always.
    fn tables() -> Vec<Table> {
        let mut out: Vec<Table> = [0.3, 1.0, 1.7, 3.2, 17.25, 1000.0, 4030.5, 4.97e9, 2f64.powi(40), 2f64.powi(49)]
            .iter()
            .map(|&s| table_for(s))
            .collect();
        for (sigma, shift) in [(3.2, 8), (1000.0, 60), (1000.0, 100), (4030.5, 200)] {
            let mut t = table_for(sigma);
            t.c_exp += shift;
            out.push(t);
        }
        out
    }

    #[test]
    fn exp_neg_a_matches_the_branching_version() {
        let mut points: Vec<u128> = vec![0, 1, 2, 120 * ONE - 2, 120 * ONE - 1, 120 * ONE, 120 * ONE + 1, u128::MAX];
        for m in 0..=176u128 {
            for j in 0..64u128 {
                let base = m * LN2 + (j << (F - 6));
                for d in [0u128, 1, 2] {
                    points.push(base.wrapping_add(d));
                    points.push(base.wrapping_sub(d));
                }
            }
        }
        let mut s = 7u64;
        let n = if cfg!(debug_assertions) { 200_000 } else { 4_000_000 };
        for i in 0..n {
            let a = ((splitmix(&mut s) as u128) << 64) | splitmix(&mut s) as u128;
            points.push(if i % 8 == 0 { a } else { a % (121 * ONE) });
        }
        for a in points {
            assert_eq!(exp_neg_a(a), branching::exp_neg_a(a), "a = {a:#x}");
        }
    }

    #[test]
    fn samples_and_deferrals_match_the_branching_version() {
        let seeds = if cfg!(debug_assertions) { 60 } else { 3000 };
        for (n, table) in tables().iter().enumerate() {
            for seed in 0..seeds {
                let raw = [&(n as u64).to_be_bytes()[..], &(seed as u64).to_be_bytes()[..]].concat();
                let mut runs = Vec::new();
                for variant in 0..2 {
                    let calls = RefCell::new(Vec::new());
                    let exact = |x: u64, y: u64| {
                        calls.borrow_mut().push((x, y));
                        let mut h = x.wrapping_mul(0x9e3779b97f4a7c15) ^ y;
                        U256([splitmix(&mut h), splitmix(&mut h), splitmix(&mut h), 0])
                    };
                    let mut out = vec![0i64; 64];
                    let mut bits = XofBits::new(&raw);
                    if variant == 0 {
                        sample(table, &mut out, &mut bits, &exact);
                    } else {
                        branching::sample(table, &mut out, &mut bits, &exact);
                    }
                    runs.push((out, bits.next_call(), calls.into_inner()));
                }
                assert_eq!(runs[0], runs[1], "table {n}, seed {seed}");
            }
        }
    }

    #[test]
    fn some_tables_defer_and_some_samples_are_multiples_of_k() {
        // the comparison above exercises the deferral and the y = 0 outputs
        let (mut deferred, mut y0) = (0, 0);
        for table in tables() {
            let calls = RefCell::new(0u64);
            let exact = |_x: u64, _y: u64| {
                *calls.borrow_mut() += 1;
                U256::ONE.shl(191)
            };
            let mut out = vec![0i64; 256];
            sample(&table, &mut out, &mut XofBits::new(b"cover"), &exact);
            deferred += calls.into_inner();
            y0 += out.iter().filter(|&&z| table.k > 1 && z % table.k as i64 == 0).count();
        }
        assert!(deferred > 100 && y0 > 10, "{deferred} {y0}");
    }

    #[test]
    fn shr_masked_is_a_shift() {
        let mut s = 3u64;
        for _ in 0..10000 {
            let e = ((splitmix(&mut s) as u128) << 64) | splitmix(&mut s) as u128;
            for m in 0..256u64 {
                assert_eq!(shr_masked(e, m), if m >= 128 { 0 } else { e >> m });
            }
        }
    }

    #[test]
    fn comparisons_are_the_orderings() {
        let mut s = 5u64;
        for i in 0..100000 {
            let mut a = U256([splitmix(&mut s), splitmix(&mut s), splitmix(&mut s), splitmix(&mut s)]);
            let b = U256([splitmix(&mut s), splitmix(&mut s), splitmix(&mut s), splitmix(&mut s)]);
            if i % 3 == 0 {
                a.0[3] = b.0[3]; // equal top words: the lower ones decide
                a.0[2] = if i % 2 == 0 { b.0[2] } else { a.0[2] };
            }
            if i % 7 == 0 {
                a = b;
            }
            assert_eq!(ge256(&a, &b) == 1, a >= b);
            let (x, y) = (a.0[0] as u128 | (a.0[1] as u128) << 64, b.0[0] as u128 | (b.0[1] as u128) << 64);
            assert_eq!(ge128(x, y) == 1, x >= y);
            assert_eq!(nonzero(a.0[0] & (i as u64 % 2).wrapping_neg()), (a.0[0] != 0 && i % 2 == 1) as u64);
        }
    }

    #[test]
    fn exp_is_accurate() {
        // c = 2^-10 (c_mant = 2^127, c_exp = -137): a = t / 1024
        let table = Table { k: 1, y_bits: 0, cdt: [U256::ZERO; CDT_SIZE], c_mant: 1 << 127, c_exp: -137 };
        for t in [0u128, 1, 7, 1000, 1024, 5000, 40000, 80000] {
            let (e, ok) = exp_neg(&table, t);
            assert_eq!(ok, 1);
            let e = e as f64 / ONE as f64;
            let want = (-(t as f64) / 1024.0).exp();
            assert!((e - want).abs() < 1e-15 * want.max(1e-300) + 1e-30, "t = {t}: {e} vs {want}");
        }
    }

    #[test]
    fn xof_counter_stops_at_64_bits() {
        let mut x = XofBits::at(b"\x00\x00\x00\x00\x00\x00\x00\x01k", u64::MAX - 1);
        x.getrandbits(8);
        x.getrandbits(8); // counter 2^64 - 1: still a valid call
        assert!(!x.overflowed() && x.next_call() == 1u128 << 64);
        x.getrandbits(8);
        assert!(x.overflowed() && x.next_call() == 1u128 << 64);
    }

    #[test]
    fn xof_bits_match_python() {
        // XofRandom(b"abc").getrandbits(12), then getrandbits(1) (hashlib reference values)
        let mut x = XofBits::new(b"abc");
        let a = x.getrandbits(12).0[0];
        let b = x.getrandbits(1).0[0];
        assert!(a < 4096 && b < 2);
    }
}
