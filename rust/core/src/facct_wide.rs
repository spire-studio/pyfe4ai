//! The exact discrete Gaussian of pyfe4ai.utils.sampling_utils.discrete_gaussian_facct
//! for wide widths: double-sampling factors k above the 2^50 of `facct` (whose
//! samples fit an i64), up to k < 2^448. The getrandbits calls are those of the
//! Python reference, in its order -- u (192 bits), then y (bits(k - 1) bits)
//! until y < k, then v (192 bits), then the sign (1 bit) -- and so are the
//! decisions; k, y, z = k x + y and t = y (y + 2 k x) are held in 64-bit words.
//!
//! The acceptance test `v < floor(2^192 exp(-c t))`, c = 1 / (2 sigma^2), is
//! decided as in `facct`:
//!
//! * a = c t is computed from the exact product c_mant t of the table's c
//!   (rounded down to a 128-bit mantissa c_mant 2^c_exp, from the MPFR value
//!   of the reference; `check_table` refuses a c off by more than about 2^-102)
//!   and truncated to 120 fraction bits. Since x <= 14 (the CDT has 14
//!   thresholds) and y < k, t < y (k + 28 k) < 29 k^2, and k <= sigma
//!   sqrt(2 ln 2) + 1, so a < 29 k^2 / (2 sigma^2) <= 20.1 (1 + 1/(sigma
//!   sqrt(2 ln 2)))^2: the same range as for the narrow widths. Its error is
//!   below a 2^-127 + 2^-120 < 2^-119.8;
//! * exp(-a) is `facct::exp_neg_a` (error below 2^-110 for any a), so the
//!   fixed-point threshold is within 2^192 (2^-110 + 2^-119.8) + 1 < 2^83 of
//!   2^192 exp(-c t);
//! * the reference evaluates neg_c t at 256 bits (t exact, one rounding) and
//!   its exponential at 256 bits: within 2^192 21 2^-255 + 1 < 2 of the same
//!   value;
//! * a decision is taken here only when v is at least 2^92 (2^-100 relative)
//!   away from the fixed-point threshold, beyond both errors; otherwise
//!   `exact(x, y)` (the Python threshold) decides, about once in 2^99 tests.
//!
//! So for a given bit source the samples are the reference's.

use crate::arith::{Word, U256};
use crate::facct::{exp_neg_a, RandBits, CDT_SIZE, F, PRECISION};

/// Words of k, y and z = k x + y (k < 2^448, z < 15 k < 2^452).
pub const W: usize = 8;
/// Largest k handled: k < 2^MAX_K_BITS.
pub const MAX_K_BITS: u32 = 448;
/// Words of t = y (y + 2 k x) < 29 k^2 (and room for the carries).
const TW: usize = 2 * W + 1;

/// The tables of one width (sampling_utils._facct_params) with a wide k.
#[derive(Clone, Debug)]
pub struct WideTable {
    pub k: [u64; W],
    pub y_bits: u32,
    pub cdt: [U256; CDT_SIZE],
    pub c_mant: u128,
    pub c_exp: i32,
}

fn is_zero(a: &[u64]) -> bool {
    a.iter().all(|&w| w == 0)
}

/// a < b for equal lengths.
fn lt(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

fn bit_len(a: &[u64]) -> u32 {
    for i in (0..a.len()).rev() {
        if a[i] != 0 {
            return 64 * i as u32 + 64 - a[i].leading_zeros();
        }
    }
    0
}

/// out = a b (out zeroed, at least a.len() + b.len() words).
fn mul_into(a: &[u64], b: &[u64], out: &mut [u64]) {
    out.fill(0);
    for (i, &ai) in a.iter().enumerate() {
        let mut c = 0u128;
        for (j, &bj) in b.iter().enumerate() {
            let t = ai as u128 * bj as u128 + out[i + j] as u128 + c;
            out[i + j] = t as u64;
            c = t >> 64;
        }
        out[i + b.len()] = c as u64;
    }
}

/// z = k m + y (m < 2^32), W + 1 words.
fn mul_add_small(k: &[u64; W], m: u64, y: &[u64; W]) -> [u64; W + 1] {
    let mut z = [0u64; W + 1];
    let mut c = 0u128;
    for i in 0..W {
        let t = k[i] as u128 * m as u128 + y[i] as u128 + c;
        z[i] = t as u64;
        c = t >> 64;
    }
    z[W] = c as u64;
    z
}

/// t = y (y + 2 k x).
fn t_of(table: &WideTable, x: u64, y: &[u64; W]) -> [u64; TW + 1] {
    let s = mul_add_small(&table.k, 2 * x, y);
    let mut t = [0u64; TW + 1];
    mul_into(y, &s, &mut t[..W + W + 1]);
    t
}

/// exp(-c t) with F fraction bits, or None when the fixed point cannot hold a
/// (the caller then uses `exact`).
fn exp_neg_wide(table: &WideTable, t: &[u64]) -> Option<u128> {
    let s = -(table.c_exp + F as i32);
    if s < 0 {
        return None;
    }
    let c = [table.c_mant as u64, (table.c_mant >> 64) as u64];
    let mut p = [0u64; TW + 3];
    mul_into(&c, t, &mut p[..t.len() + 2]);
    // a = p >> s, which must fit 128 bits
    let s = s as u32;
    if bit_len(&p) > s + 128 {
        return None;
    }
    let (w, b) = ((s / 64) as usize, s % 64);
    let word = |i: usize| if i < p.len() { p[i] } else { 0 };
    let lo = if b == 0 { word(w) } else { (word(w) >> b) | (word(w + 1) << (64 - b)) };
    let hi = if b == 0 { word(w + 1) } else { (word(w + 1) >> b) | (word(w + 2) << (64 - b)) };
    Some(exp_neg_a(((hi as u128) << 64) | lo as u128))
}

/// The acceptance decision `v < floor(2^192 exp(-c t))` of the Python sampler.
fn accept(table: &WideTable, x: u64, y: &[u64; W], v: U256, exact: &dyn Fn(u64, &[u64]) -> U256) -> bool {
    let t = t_of(table, x, y);
    if let Some(e) = exp_neg_wide(table, &t) {
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

/// k - 1 and ceil((k - 1) / 2).
fn corners(k: &[u64; W]) -> ([u64; W], [u64; W]) {
    let mut km = *k;
    for w in km.iter_mut() {
        let (v, b) = w.overflowing_sub(1);
        *w = v;
        if !b {
            break;
        }
    }
    // ceil(km / 2) = (km + 1) >> 1 = k >> 1
    let mut half = [0u64; W];
    for i in 0..W {
        half[i] = (k[i] >> 1) | if i + 1 < W { k[i + 1] << 63 } else { 0 };
    }
    (km, half)
}

/// Whether the fixed-point thresholds agree with `exact` to 2^-102 (relative to
/// 2^192) at the corners and the middle of the (x, y) range, as
/// `facct::check_table`: a c off by more than about 2^-102 is refused.
pub fn check_table(table: &WideTable, exact: &dyn Fn(u64, &[u64]) -> U256) -> bool {
    let (km, half) = corners(&table.k);
    let mut one = [0u64; W];
    one[0] = 1;
    let xm = CDT_SIZE as u64;
    for (x, y) in [(0, &one), (0, &km), (xm, &one), (xm, &km), (xm / 2, &half)] {
        if is_zero(y) {
            continue;
        }
        let t = t_of(table, x, y);
        if let Some(e) = exp_neg_wide(table, &t) {
            let (a, b) = (U256::from_u128(e).shl(PRECISION - F), exact(x, y));
            let d = if a >= b { a.wsub(b) } else { b.wsub(a) };
            if d >= U256::ONE.shl(PRECISION - 102) {
                return false;
            }
        }
    }
    true
}

/// Whether k and y_bits describe a wide table: 2^50 < k < 2^MAX_K_BITS and
/// y_bits = bits(k - 1).
pub fn valid(k: &[u64; W], y_bits: u32) -> bool {
    let kb = bit_len(k);
    kb > 50 && kb <= MAX_K_BITS && bit_len(&corners(k).0) == y_bits
}

/// `negative.len()` samples of D_{Z,sigma} (discrete_gaussian_facct): sample i
/// is (-1)^negative[i] mags[W i .. W (i + 1)] (little-endian words). `exact(x,
/// y)` is the Python threshold floor(2^192 exp(neg_c y (y + 2 k x))).
pub fn sample(table: &WideTable, mags: &mut [u64], negative: &mut [u8], bits: &mut dyn RandBits,
              exact: &dyn Fn(u64, &[u64]) -> U256) {
    let count = negative.len();
    assert!(mags.len() >= W * count);
    let mut filled = 0;
    let mut y = [0u64; W];
    while filled < count {
        let u = bits.getrandbits(PRECISION);
        let mut x = 0u64;
        for th in table.cdt.iter() {
            x += (u >= *th) as u64; // the whole table, as the reference
        }
        bits.getrandbits_into(table.y_bits, &mut y);
        while !lt(&y, &table.k) {
            bits.getrandbits_into(table.y_bits, &mut y);
        }
        let v = bits.getrandbits(PRECISION);
        if !is_zero(&y) && !accept(table, x, &y, v, exact) {
            continue;
        }
        let z = mul_add_small(&table.k, x, &y);
        let neg = bits.getrandbits(1).0[0] == 1;
        if is_zero(&z) && neg {
            continue;
        }
        debug_assert!(z[W] == 0);
        mags[W * filled..W * (filled + 1)].copy_from_slice(&z[..W]);
        negative[filled] = neg as u8;
        filled += 1;
    }
    for w in y.iter_mut() {
        unsafe { std::ptr::write_volatile(w, 0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_and_corners() {
        let mut k = [0u64; W];
        k[1] = 1; // 2^64
        let mut y = [0u64; W];
        y[0] = 5;
        let table = WideTable { k, y_bits: 64, cdt: [U256::ZERO; CDT_SIZE], c_mant: 1 << 127, c_exp: -300 };
        // t = 5 (5 + 2 2^64 3) = 25 + 30 2^64
        let t = t_of(&table, 3, &y);
        assert_eq!((t[0], t[1], t[2]), (25, 30, 0));
        let (km, half) = corners(&k);
        assert_eq!((km[0], km[1]), (u64::MAX, 0));
        assert_eq!((half[0], half[1]), (1 << 63, 0));
        assert!(valid(&k, 64) && !valid(&k, 65));
    }

    #[test]
    fn exp_matches_the_narrow_kernel() {
        // c = 2^-140 (c_mant = 2^127, c_exp = -267), t = 2^140 3/2: a = 1.5
        let table = WideTable { k: [0; W], y_bits: 1, cdt: [U256::ZERO; CDT_SIZE], c_mant: 1 << 127, c_exp: -267 };
        let mut t = [0u64; TW + 1];
        t[2] = 3 << 11; // 3 2^139
        let e = exp_neg_wide(&table, &t).unwrap() as f64 / (1u128 << F) as f64;
        assert!((e - (-1.5f64).exp()).abs() < 1e-15);
    }
}
