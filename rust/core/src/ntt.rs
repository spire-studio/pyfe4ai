//! Negacyclic NTT of Z_q[x] / (x^n + 1) with the root, tables and butterfly
//! order of pyfe4ai.utils.ring_lwe_utils.ntt_forward / ntt_inverse: the values
//! are the Python ones in both domains. Shoup twiddles, Harvey's lazy
//! butterflies (values in [0, 4q), which needs 4q < 2^BITS).

use crate::arith::{Modulus, Word};

#[derive(Clone, Debug)]
pub struct Ring<W: Word> {
    pub m: Modulus<W>,
    pub n: usize,
    pub logn: u32,
    pub q: u128,
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
    pub fn new(q: u128, n: usize, root: u128) -> Result<Self, &'static str> {
        if n < 2 || !n.is_power_of_two() || n > (1 << 24) {
            return Err("n must be a power of two in [2, 2^24]");
        }
        let limit = if W::BITS == 64 { 1u128 << 62 } else { 1u128 << 125 };
        if q < 3 || q & 1 == 0 || q >= limit || root >= q || (q - 1) % (2 * n as u128) != 0 {
            return Err("unsupported (q, n): need an odd q = 1 mod 2n below the word limit");
        }
        let m = Modulus::<W>::new(q);
        let rt = W::from_u128(root);
        let root_inv = m.pow(rt, (2 * n - 1) as u128);
        if m.pow(rt, n as u128) != W::from_u128(q - 1) || m.mulp(rt, root_inv) != W::ONE {
            return Err("root is not a primitive 2n-th root of unity mod q");
        }
        let nw = W::from_u128(n as u128 % q);
        let n_inv = m.pow(nw, q - 2); // q is prime
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
                let step = m.pow(w2, (n / len) as u128);
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
}

impl AnyRing {
    pub fn new(q: u128, n: usize, root: u128) -> Result<Self, &'static str> {
        if q >= (1u128 << 125) {
            return Err("q must be below 2^125");
        }
        if q < (1u128 << 62) {
            Ok(AnyRing::W64(Ring::new(q, n, root)?))
        } else {
            Ok(AnyRing::W128(Ring::new(q, n, root)?))
        }
    }
    pub fn word_bits(&self) -> u32 {
        match self {
            AnyRing::W64(_) => 64,
            AnyRing::W128(_) => 128,
        }
    }
    pub fn n(&self) -> usize {
        match self {
            AnyRing::W64(r) => r.n,
            AnyRing::W128(r) => r.n,
        }
    }
    pub fn q(&self) -> u128 {
        match self {
            AnyRing::W64(r) => r.q,
            AnyRing::W128(r) => r.q,
        }
    }
    pub fn qbits(&self) -> u32 {
        128 - self.q().leading_zeros()
    }
}

/// Dispatch on the word size: `with_ring!(ring, r => expr)`.
#[macro_export]
macro_rules! with_ring {
    ($ring:expr, $r:ident => $body:expr) => {
        match $ring {
            $crate::ntt::AnyRing::W64($r) => $body,
            $crate::ntt::AnyRing::W128($r) => $body,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn primitive_root(q: u128, n: usize) -> u128 {
        let order = 2 * n as u128;
        for g in 2u128..1000 {
            let m = Modulus::<u128>::new(q);
            let r = m.pow(g, (q - 1) / order);
            if r != 1 && m.pow(r, order) == 1 && m.pow(r, order / 2) != 1 {
                return r;
            }
        }
        panic!("no root")
    }

    fn is_prime(n: u128) -> bool {
        if n < 2 {
            return false;
        }
        let m = Modulus::<u128>::new(if n & 1 == 1 { n } else { 3 });
        if n & 1 == 0 {
            return n == 2;
        }
        let (mut d, mut s) = (n - 1, 0);
        while d & 1 == 0 {
            d >>= 1;
            s += 1;
        }
        'outer: for &a in &[2u128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if a % n == 0 {
                continue;
            }
            let mut x = m.pow(a, d);
            if x == 1 || x == n - 1 {
                continue;
            }
            for _ in 1..s {
                x = m.mulp(x, x);
                if x == n - 1 {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    pub fn ntt_prime(bits: u32, n: usize) -> u128 {
        let m = 2 * n as u128;
        let mut c = (1u128 << (bits - 1)) / m * m + 1;
        while !is_prime(c) {
            c += m;
        }
        c
    }

    fn schoolbook(a: &[u128], b: &[u128], q: u128) -> Vec<u128> {
        let n = a.len();
        let m = Modulus::<u128>::new(q);
        let mut out = vec![0u128; n];
        for i in 0..n {
            for j in 0..n {
                let p = m.mulp(a[i], b[j]);
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
        for &(bits, n) in &[(20u32, 8usize), (40, 16), (61, 64), (62, 32), (77, 64), (124, 32)] {
            let q = ntt_prime(bits, n);
            let root = primitive_root(q, n);
            let ring = AnyRing::new(q, n, root).unwrap();
            let mut s = q ^ 0xdead_beef;
            let mut rnd = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 7) % q
            };
            let a: Vec<u128> = (0..n).map(|_| rnd()).collect();
            let b: Vec<u128> = (0..n).map(|_| rnd()).collect();
            let want = schoolbook(&a, &b, q);
            with_ring!(&ring, r => {
                let aw: Vec<_> = a.iter().map(|&v| Word::from_u128(v)).collect();
                let bw: Vec<_> = b.iter().map(|&v| Word::from_u128(v)).collect();
                let got: Vec<u128> = r.poly_mul(&aw, &bw).iter().map(|v| v.to_u128()).collect();
                assert_eq!(got, want, "bits {} n {}", bits, n);
                let mut f = aw.clone();
                let mut back = aw.clone();
                r.forward(&aw, &mut f);
                r.inverse(&f, &mut back);
                assert_eq!(back, aw);
            });
        }
    }
}
