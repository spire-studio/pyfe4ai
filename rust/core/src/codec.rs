//! Decimal strings (the JSON ciphertext format: gmpy2.digits out, int(s) % q in)
//! and the binary ciphertext format 1 of ring_lwe_utils.encode_binary_ciphertext.
//! The u128 functions are the u64 / u128 words' `Word::parse_dec` / `format_dec`.

use crate::arith::Word;

/// Fast path of int(s) % q: optional '-', then 1..38 ASCII digits; None: use Python.
#[inline]
pub fn parse_dec(s: &[u8], q: u128) -> Option<u128> {
    let neg = !s.is_empty() && s[0] == b'-';
    let d = if neg { &s[1..] } else { s };
    if d.is_empty() || d.len() > 38 {
        return None;
    }
    let head = d.len().saturating_sub(19);
    let (mut hi, mut lo) = (0u64, 0u64);
    for &c in &d[..head] {
        let v = c.wrapping_sub(b'0');
        if v > 9 {
            return None;
        }
        hi = hi * 10 + v as u64;
    }
    for &c in &d[head..] {
        let v = c.wrapping_sub(b'0');
        if v > 9 {
            return None;
        }
        lo = lo * 10 + v as u64;
    }
    let mut v = if head > 0 { hi as u128 * 10_000_000_000_000_000_000u128 + lo as u128 } else { lo as u128 };
    if v >= q {
        v %= q;
    }
    Some(if neg && v != 0 { q - v } else { v })
}

const DIGITS2: &[u8; 200] = b"0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

/// Decimal digits of v < 2^127 appended to `out`.
pub fn format_dec(v: u128, out: &mut Vec<u8>) {
    let mut tmp = [0u8; 40];
    let mut p = tmp.len();
    let mut lo: u64;
    if v >> 64 != 0 {
        const TEN19: u64 = 10_000_000_000_000_000_000;
        let hi = v / TEN19 as u128;
        let mut rem = (v - hi * TEN19 as u128) as u64;
        for _ in 0..9 {
            let r = (rem % 100) as usize;
            rem /= 100;
            p -= 2;
            tmp[p..p + 2].copy_from_slice(&DIGITS2[2 * r..2 * r + 2]);
        }
        p -= 1;
        tmp[p] = b'0' + rem as u8;
        lo = hi as u64;
        if lo == 0 {
            out.extend_from_slice(&tmp[p..]);
            return;
        }
    } else {
        lo = v as u64;
        if lo == 0 {
            out.push(b'0');
            return;
        }
    }
    while lo >= 100 {
        let r = (lo % 100) as usize;
        lo /= 100;
        p -= 2;
        tmp[p..p + 2].copy_from_slice(&DIGITS2[2 * r..2 * r + 2]);
    }
    if lo >= 10 {
        p -= 2;
        tmp[p..p + 2].copy_from_slice(&DIGITS2[2 * lo as usize..2 * lo as usize + 2]);
    } else {
        p -= 1;
        tmp[p] = b'0' + lo as u8;
    }
    out.extend_from_slice(&tmp[p..]);
}

/// Exactly 19 decimal digits of v < 10^19 (leading zeros kept) appended to `out`.
pub fn format_dec19(mut v: u64, out: &mut Vec<u8>) {
    let mut tmp = [0u8; 19];
    let mut p = 19;
    for _ in 0..9 {
        let r = (v % 100) as usize;
        v /= 100;
        p -= 2;
        tmp[p..p + 2].copy_from_slice(&DIGITS2[2 * r..2 * r + 2]);
    }
    tmp[0] = b'0' + v as u8;
    out.extend_from_slice(&tmp);
}

pub const BIN_HEADER: usize = 10;

/// Bytes per coefficient of the binary format: ceil(bits(q) / 8).
pub fn bin_wb(qbits: u32) -> usize {
    ((qbits + 7) / 8) as usize
}

pub fn bin_size(eta: usize, n: usize, wb: usize) -> usize {
    BIN_HEADER + (eta + 1) * n * wb
}

/// "PFRL", version 1, wb, eta (u16 LE), log2 n, 0; then the eta rows of ct0 and ct1.
pub fn bin_write<W: Word>(qbits: u32, logn: u32, eta: usize, ct0: &[W], ct1: &[W], out: &mut Vec<u8>) {
    let wb = bin_wb(qbits);
    out.extend_from_slice(b"PFRL");
    out.extend_from_slice(&[1, wb as u8, (eta & 0xFF) as u8, (eta >> 8) as u8, logn as u8, 0]);
    let start = out.len();
    out.resize(start + (ct0.len() + ct1.len()) * wb, 0);
    for (&v, o) in ct0.iter().chain(ct1.iter()).zip(out[start..].chunks_exact_mut(wb)) {
        v.write_le(o);
    }
}

/// The coefficients of a format-1 ciphertext reduced mod q; None if the header
/// or the length does not match (eta, n, q).
pub fn bin_read<W: Word>(data: &[u8], q: W, logn: u32, eta: usize, out: &mut [W]) -> Option<()> {
    let wb = bin_wb(q.bits());
    let n = 1usize << logn;
    if data.len() != bin_size(eta, n, wb)
        || &data[..4] != b"PFRL"
        || data[4] != 1
        || data[5] as usize != wb
        || (data[6] as usize | (data[7] as usize) << 8) != eta
        || data[8] as u32 != logn
        || data[9] != 0
        || out.len() != (eta + 1) * n
    {
        return None;
    }
    for (o, c) in out.iter_mut().zip(data[BIN_HEADER..].chunks_exact(wb)) {
        let v = W::read_le(c);
        *o = if v < q { v } else { v.rem(q) };
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_round_trip() {
        let q = (1u128 << 124) + 0x9001;
        for v in [0u128, 1, 9, 10, 99, 100, 12345678901234567890, (1u128 << 64) - 1, 1u128 << 64, q - 1] {
            let mut s = Vec::new();
            format_dec(v, &mut s);
            assert_eq!(String::from_utf8(s.clone()).unwrap(), v.to_string());
            assert_eq!(parse_dec(&s, q), Some(v % q));
        }
        assert_eq!(parse_dec(b"-5", 7), Some(2));
        assert_eq!(parse_dec(b"-0", 7), Some(0));
        for bad in [&b""[..], b"-", b"+5", b" 5", b"1_0", b"12a"] {
            assert_eq!(parse_dec(bad, 7), None);
        }
        let mut s = Vec::new();
        format_dec19(42, &mut s);
        assert_eq!(s, b"0000000000000000042");
    }

    #[test]
    fn wide_decimal_and_binary() {
        use crate::arith::U256;
        let q = U256::parse_dec(b"3291009114642412084309938365114701009965471731267159726697299969", U256::MAX).unwrap();
        let qm1 = q.wsub(U256::ONE);
        let mut s = Vec::new();
        qm1.format_dec(&mut s);
        assert_eq!(s, b"3291009114642412084309938365114701009965471731267159726697299968");
        assert_eq!(U256::parse_dec(&s, q), Some(qm1));
        assert_eq!(U256::parse_dec(b"-1", q), Some(qm1));
        assert_eq!(U256::parse_dec(b"-0", q), Some(U256::ZERO));
        // q + 5 and 10^76 reduce mod q
        let mut t = Vec::new();
        q.wadd(U256::from_u128(5)).format_dec(&mut t);
        assert_eq!(U256::parse_dec(&t, q), Some(U256::from_u128(5)));
        let big = [b'1'].iter().chain([b'0'; 76].iter()).copied().collect::<Vec<u8>>();
        let ten76 = U256::parse_dec(&big, U256::MAX).unwrap();
        assert_eq!(U256::parse_dec(&big, q), Some(ten76.rem(q)));
        assert_eq!(U256::parse_dec(&[b'9'; 78], q), None); // >= 2^256: Python's
        assert_eq!(U256::parse_dec(&[b'9'; 79], q), None);
        for bad in [&b""[..], b"-", b"+5", b" 5", b"12a"] {
            assert_eq!(U256::parse_dec(bad, q), None);
        }
        let ct0 = [U256::from_u128(1), qm1];
        let ct1 = [U256::from_u128(7), U256::ONE.shl(200)];
        let mut out = Vec::new();
        bin_write(q.bits(), 1, 1, &ct0, &ct1, &mut out);
        assert_eq!(out.len(), bin_size(1, 2, 27));
        let mut back = [U256::ZERO; 4];
        assert!(bin_read(&out, q, 1, 1, &mut back).is_some());
        assert_eq!(back, [ct0[0], ct0[1], ct1[0], ct1[1]]);
    }
}
