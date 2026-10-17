//! Negacyclic NTT of Z_q[x] / (x^n + 1) with the root, tables and butterfly
//! order of pyfe4ai.utils.ring_lwe_utils.ntt_forward / ntt_inverse: the values
//! are the Python ones in both domains. Shoup twiddles, Harvey's lazy
//! butterflies (values in [0, 4q), which needs 4q < 2^BITS: q < 2^MAX_Q_BITS).

use crate::arith::{Modulus, Word, U256};

#[derive(Clone, Debug)]
pub struct Ring<W: Word> {
    pub m: Modulus<W>,
    pub n: usize,
    pub logn: u32,
    pub q: W,
    br: Vec<u32>,
    twist: Vec<W>,
    twist_p: Vec<W>,
    scale: Vec<W>,
    scale_p: Vec<W>,
    fwd: Vec<W>,
    fwd_p: Vec<W>,
    inv: Vec<W>,
    inv_p: Vec<W>,
}

impl<W: Word> Ring<W> {
    /// The ring for an NTT prime q (q = 1 mod 2n) and a primitive 2n-th root of unity.
    pub fn new(q: W, n: usize, root: W) -> Result<Self, &'static str> {
        if n < 2 || !n.is_power_of_two() || n > (1 << 24) {
            return Err("n must be a power of two in [2, 2^24]");
        }
        if q < W::from_u128(3) || !q.bit0() || q.bits() > W::MAX_Q_BITS || root >= q
            || q.wsub(W::ONE).rem_u64(2 * n as u64) != 0
        {
            return Err("unsupported (q, n): need an odd q = 1 mod 2n below the word limit");
        }
        let m = Modulus::<W>::new(q);
        let rt = root;
        let small = |v: usize| W::from_u128(v as u128);
        let root_inv = m.pow(rt, small(2 * n - 1));
        if m.pow(rt, small(n)) != q.wsub(W::ONE) || m.mulp(rt, root_inv) != W::ONE {
            return Err("root is not a primitive 2n-th root of unity mod q");
        }
        let nw = m.reduce(small(n));
        let n_inv = m.pow(nw, q.wsub(W::from_u128(2))); // q is prime
        if m.mulp(nw, n_inv) != W::ONE {
            return Err("n is not invertible mod q");
        }
        let logn = n.trailing_zeros();
        let mut r = Ring {
            m: m.clone(),
            n,
            logn,
            q,
            br: Vec::with_capacity(n),
            twist: Vec::with_capacity(n),
            twist_p: Vec::with_capacity(n),
            scale: Vec::with_capacity(n),
            scale_p: Vec::with_capacity(n),
            fwd: Vec::new(),
            fwd_p: Vec::new(),
            inv: Vec::new(),
            inv_p: Vec::new(),
        };
        let (mut t, mut u) = (W::ONE, W::ONE);
        for i in 0..n {
            r.twist.push(t);
            r.twist_p.push(m.shoup_quot(t));
            let s = m.mulp(n_inv, u);
            r.scale.push(s);
            r.scale_p.push(m.shoup_quot(s));
            t = m.mulp(t, rt);
            u = m.mulp(u, root_inv);
            r.br.push((i as u32).reverse_bits() >> (32 - logn));
        }
        let stages = |base: W| -> (Vec<W>, Vec<W>) {
            let w2 = m.mulp(base, base);
            let mut out = Vec::with_capacity(n);
            let mut len = 2;
            while len <= n {
                let step = m.pow(w2, small(n / len));
                let mut w = W::ONE;
                out.push(W::ONE);
                for _ in 1..len / 2 {
                    w = m.mulp(w, step);
                    out.push(w);
                }
                len <<= 1;
            }
            let p = out.iter().map(|&v| m.shoup_quot(v)).collect();
            (out, p)
        };
        let (f, fp) = stages(rt);
        let (iv, ivp) = stages(root_inv);
        r.fwd = f;
        r.fwd_p = fp;
        r.inv = iv;
        r.inv_p = ivp;
        Ok(r)
    }

    fn butterflies(&self, x: &mut [W], tw: &[W], twp: &[W]) {
        let q = self.m.q;
        let q2 = q.wadd(q);
        let n = self.n;
        let mut off = 0;
        let mut len = 2;
        while len <= n {
            let half = len >> 1;
            let (t, tp) = (&tw[off..off + half], &twp[off..off + half]);
            for blk in x.chunks_exact_mut(len) {
                let (lo, hi) = blk.split_at_mut(half);
                for j in 0..half {
                    let mut u = lo[j];
                    if u >= q2 {
                        u = u.wsub(q2);
                    }
                    let v = self.m.shoup(hi[j], t[j], tp[j]);
                    lo[j] = u.wadd(v);
                    hi[j] = u.wsub(v).wadd(q2);
                }
            }
            off += half;
            len <<= 1;
        }
    }

    #[inline(always)]
    fn reduce4(&self, v: W) -> W {
        let q = self.m.q;
        let q2 = q.wadd(q);
        let v = if v >= q2 { v.wsub(q2) } else { v };
        if v >= q { v.wsub(q) } else { v }
    }

    /// ring_lwe_utils.ntt_forward: twist, bit-reverse, stages; out in [0, q).
    pub fn forward(&self, input: &[W], out: &mut [W]) {
        for i in 0..self.n {
            let b = self.br[i] as usize;
            out[i] = self.m.shoup(input[b], self.twist[b], self.twist_p[b]);
        }
        self.butterflies(out, &self.fwd, &self.fwd_p);
        for v in out.iter_mut() {
            *v = self.reduce4(*v);
        }
    }

    /// ring_lwe_utils.ntt_inverse: bit-reverse, stages, times n^-1 untwist[i]; out in [0, q).
    pub fn inverse(&self, input: &[W], out: &mut [W]) {
        for i in 0..self.n {
            out[i] = input[self.br[i] as usize];
        }
        self.butterflies(out, &self.inv, &self.inv_p);
        let q = self.m.q;
        for i in 0..self.n {
            let v = self.m.shoup(out[i], self.scale[i], self.scale_p[i]);
            out[i] = if v >= q { v.wsub(q) } else { v };
        }
    }

    /// Forward NTT in Montgomery form (the key format of encryption / decryption).
    pub fn forward_mont(&self, input: &[W], out: &mut [W]) {
        self.forward(input, out);
        for v in out.iter_mut() {
            *v = self.m.to_mont(*v);
        }
    }

    /// The negacyclic product (for tests).
    pub fn poly_mul(&self, a: &[W], b: &[W]) -> Vec<W> {
        let (mut fa, mut fb, mut out) = (vec![W::ZERO; self.n], vec![W::ZERO; self.n], vec![W::ZERO; self.n]);
        self.forward(a, &mut fa);
        self.forward(b, &mut fb);
        for i in 0..self.n {
            fa[i] = self.m.mulp(fa[i], fb[i]);
        }
        self.inverse(&fa, &mut out);
        out
    }
}

/// A ring in the word size its modulus needs.
#[derive(Clone, Debug)]
pub enum AnyRing {
    W64(Ring<u64>),
    W128(Ring<u128>),
    W256(Ring<U256>),
}

impl AnyRing {
    /// u64 words for q < 2^62, u128 words for q < 2^125, 256-bit words for q < 2^253.
    pub fn new(q: U256, n: usize, root: U256) -> Result<Self, &'static str> {
        let bits = q.bits();
        if bits > U256::MAX_Q_BITS {
            return Err("q must be below 2^253");
        }
        if root >= q {
            return Err("unsupported (q, n): need an odd q = 1 mod 2n below the word limit");
        }
        if bits <= u64::MAX_Q_BITS {
            Ok(AnyRing::W64(Ring::new(u64::from_u256(q), n, u64::from_u256(root))?))
        } else if bits <= u128::MAX_Q_BITS {
            Ok(AnyRing::W128(Ring::new(u128::from_u256(q), n, u128::from_u256(root))?))
        } else {
            Ok(AnyRing::W256(Ring::new(q, n, root)?))
        }
    }
    pub fn word_bits(&self) -> u32 {
        match self {
            AnyRing::W64(_) => 64,
            AnyRing::W128(_) => 128,
            AnyRing::W256(_) => 256,
        }
    }
    pub fn n(&self) -> usize {
        match self {
            AnyRing::W64(r) => r.n,
            AnyRing::W128(r) => r.n,
            AnyRing::W256(r) => r.n,
        }
    }
    pub fn q(&self) -> U256 {
        match self {
            AnyRing::W64(r) => r.q.to_u256(),
            AnyRing::W128(r) => r.q.to_u256(),
            AnyRing::W256(r) => r.q,
        }
    }
    pub fn qbits(&self) -> u32 {
        self.q().bits()
    }
}

/// Dispatch on the word size: `with_ring!(ring, r => expr)`.
#[macro_export]
macro_rules! with_ring {
    ($ring:expr, $r:ident => $body:expr) => {
        match $ring {
            $crate::ntt::AnyRing::W64($r) => $body,
            $crate::ntt::AnyRing::W128($r) => $body,
            $crate::ntt::AnyRing::W256($r) => $body,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::tests::{slow_mulmod, u256, Lcg};

    fn primitive_root<W: Word>(q: W, n: usize) -> W {
        let order = 2 * n as u64;
        let m = Modulus::<W>::new(q);
        let e = q.wsub(W::ONE);
        // (q - 1) / order by repeated halving (order is a power of two)
        let mut cof = e;
        for _ in 0..order.trailing_zeros() {
            cof = cof.shr1();
        }
        for g in 2u128..1000 {
            let r = m.pow(W::from_u128(g), cof);
            if m.pow(r, W::from_u128(order as u128 / 2)) != W::ONE {
                return r;
            }
        }
        panic!("no root")
    }

    fn is_prime<W: Word>(n: W) -> bool {
        if n < W::from_u128(4) {
            return n > W::ONE;
        }
        if !n.bit0() {
            return false;
        }
        let m = Modulus::<W>::new(n);
        let nm1 = n.wsub(W::ONE);
        let (mut d, mut s) = (nm1, 0);
        while !d.bit0() {
            d = d.shr1();
            s += 1;
        }
        'outer: for &a in &[2u128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            let aw = W::from_u128(a);
            if aw.rem(n) == W::ZERO {
                continue;
            }
            let mut x = m.pow(aw, d);
            if x == W::ONE || x == nm1 {
                continue;
            }
            for _ in 1..s {
                x = m.mulp(x, x);
                if x == nm1 {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    pub fn ntt_prime(bits: u32, n: usize) -> U256 {
        let m = U256::from_u128(2 * n as u128);
        let base = U256::ONE.shl(bits - 1);
        // the first c = 1 mod 2n above 2^(bits - 1)
        let mut c = base.wsub(base.rem(m)).wadd(U256::ONE);
        if c < base {
            c = c.wadd(m);
        }
        while !is_prime(c) {
            c = c.wadd(m);
        }
        c
    }

    fn schoolbook(a: &[U256], b: &[U256], q: U256) -> Vec<U256> {
        let n = a.len();
        let m = Modulus::<U256>::new(q);
        let mut out = vec![U256::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                let p = slow_mulmod(a[i], b[j], q);
                let k = i + j;
                if k < n {
                    out[k] = m.add(out[k], p);
                } else {
                    out[k - n] = m.sub(out[k - n], p);
                }
            }
        }
        out
    }

    #[test]
    fn ntt_products_match_schoolbook() {
        for &(bits, n) in &[(20u32, 8usize), (40, 16), (61, 64), (62, 32), (77, 64), (124, 32), (125, 16),
                            (126, 16), (160, 32), (212, 64), (252, 16), (253, 32)] {
            let q = ntt_prime(bits, n);
            assert_eq!(q.bits(), bits);
            let root = primitive_root(q, n);
            let ring = AnyRing::new(q, n, root).unwrap();
            let want_bits = if bits <= 62 { 64 } else if bits <= 125 { 128 } else { 256 };
            assert_eq!(ring.word_bits(), want_bits);
            let mut rng = Lcg(q.0[0] as u128 ^ 0xdead_beef);
            let a: Vec<U256> = (0..n).map(|_| rng.below(q)).collect();
            let b: Vec<U256> = (0..n).map(|_| rng.below(q)).collect();
            let want = schoolbook(&a, &b, q);
            with_ring!(&ring, r => {
                let aw: Vec<_> = a.iter().map(|&v| Word::from_u256(v)).collect();
                let bw: Vec<_> = b.iter().map(|&v| Word::from_u256(v)).collect();
                let got: Vec<U256> = r.poly_mul(&aw, &bw).iter().map(|v| v.to_u256()).collect();
                assert_eq!(got, want, "bits {} n {}", bits, n);
                let mut f = aw.clone();
                let mut back = aw.clone();
                r.forward(&aw, &mut f);
                r.inverse(&f, &mut back);
                assert_eq!(back, aw);
            });
            if bits <= 125 {
                // the 256-bit word gives the same transforms for small moduli
                let wide = Ring::<U256>::new(q, n, root).unwrap();
                let mut f = a.clone();
                wide.forward(&a, &mut f);
                with_ring!(&ring, r => {
                    let aw: Vec<_> = a.iter().map(|&v| Word::from_u256(v)).collect();
                    let mut g = aw.clone();
                    r.forward(&aw, &mut g);
                    assert_eq!(g.iter().map(|v| v.to_u256()).collect::<Vec<_>>(), f);
                });
            }
        }
        assert!(AnyRing::new(ntt_prime(254, 16), 16, U256::from_u128(3)).is_err());
        assert_eq!(u256("97"), U256::from_u128(97));
    }
}
