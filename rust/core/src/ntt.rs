//! Negacyclic NTT of Z_q[x] / (x^n + 1) with the root, tables and butterfly
//! order of pyfe4ai.primitives.lattice.ring_lwe_utils.ntt_forward / ntt_inverse: the values
//! are the Python ones in both domains. Shoup twiddles, Harvey's lazy
//! butterflies (values in [0, 4q)) when 4q < 2^BITS, else butterflies that keep
//! the values in [0, q) (u64 words for 2^62 <= q < 2^63).

use crate::arith::{Modulus, Word, U256};

#[derive(Clone, Debug)]
pub struct Ring<W: Word> {
    pub m: Modulus<W>,
    pub n: usize,
    pub logn: u32,
    pub q: W,
    /// 4q < 2^BITS: Harvey's lazy butterflies.
    lazy: bool,
    br: Vec<u32>,
    twist: Vec<W>,
    twist_p: Vec<W>,
    scale: Vec<W>,
    scale_p: Vec<W>,
    fwd: Vec<W>,
    fwd_p: Vec<W>,
    inv: Vec<W>,
    inv_p: Vec<W>,
    /// For a composite q = p_0 ... p_{k-1} (Ring::new_rns): the NTT per prime.
    pub rns: Option<Box<Rns>>,
}

/// The RNS backend of a ring for a composite q = p_0 ... p_{k-1}: distinct NTT
/// primes p_j = 1 mod 2n below 2^62, k <= BITS / 64, and the residues of the
/// CRT root (ring_lwe_utils._get_ntt_root). Coefficients stay standard values
/// in [0, q); an NTT-domain value is a word whose 64-bit limb j is the residue
/// mod p_j of the NTT mod q (the negacyclic NTT mod q is the CRT of the
/// primes' NTTs), so products in the NTT domain are k 64-bit products.
#[derive(Clone, Debug)]
pub struct Rns {
    pub rings: Vec<Ring<u64>>,
    /// 2^(64 i) mod p_j (i = 0..3) and Shoup quotients, per prime.
    limb: Vec<[(u64, u64); 4]>,
    /// inv[j][i] = p_i^-1 mod p_j (i < j) and Shoup quotient (Garner).
    inv: Vec<Vec<(u64, u64)>>,
}

impl Rns {
    /// x mod p_j of a 256-bit value (its limbs).
    #[inline(always)]
    fn residue(&self, j: usize, l: &[u64; 4]) -> u64 {
        let m = &self.rings[j].m;
        let c = &self.limb[j];
        let p = m.q;
        let red = |x: u64| {
            let x = if x >= 2 * p { x - 2 * p } else { x };
            if x >= p { x - p } else { x }
        };
        // each Shoup product is below 2p < 2^63
        let a = red(m.shoup(l[0], c[0].0, c[0].1) + m.shoup(l[1], c[1].0, c[1].1));
        let b = red(m.shoup(l[2], c[2].0, c[2].1) + m.shoup(l[3], c[3].0, c[3].1));
        m.add(a, b)
    }

    /// The value in [0, q) of residues r (Garner's mixed radix).
    #[inline(always)]
    fn join(&self, r: &[u64]) -> U256 {
        let k = self.rings.len();
        let mut v = [0u64; 4];
        for j in 0..k {
            let m = &self.rings[j].m;
            let p = m.q;
            let mut t = r[j];
            for i in 0..j {
                let vi = if v[i] >= p { v[i] % p } else { v[i] };
                let (c, cp) = self.inv[j][i];
                let d = m.sub(t, vi);
                let x = m.shoup(d, c, cp);
                t = if x >= p { x - p } else { x };
            }
            v[j] = t;
        }
        let mut acc = U256::from_u128(v[k - 1] as u128);
        for i in (0..k - 1).rev() {
            acc = acc.mul_small_add(self.rings[i].m.q, v[i]).0; // < q: no carry
        }
        acc
    }

    fn forward<W: Word>(&self, input: &[W], out: &mut [W], mont: bool) {
        let n = input.len();
        let k = self.rings.len();
        let mut res = vec![0u64; k * n];
        let mut t = vec![0u64; n];
        for (i, x) in input.iter().enumerate() {
            let l = x.to_u256().0;
            for j in 0..k {
                res[j * n + i] = self.residue(j, &l);
            }
        }
        for (j, rg) in self.rings.iter().enumerate() {
            t.copy_from_slice(&res[j * n..(j + 1) * n]);
            if mont {
                rg.forward_mont(&t, &mut res[j * n..(j + 1) * n]);
            } else {
                rg.forward(&t, &mut res[j * n..(j + 1) * n]);
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            let mut limbs = [0u64; 4];
            for j in 0..k {
                limbs[j] = res[j * n + i];
            }
            *o = W::from_u256(U256(limbs));
        }
        // the input may be secret (key shares, r)
        crate::rlwe::wipe_words(&mut res);
        crate::rlwe::wipe_words(&mut t);
    }

    fn inverse<W: Word>(&self, input: &[W], out: &mut [W]) {
        let n = input.len();
        let k = self.rings.len();
        let mut res = vec![0u64; k * n];
        let mut t = vec![0u64; n];
        for (i, x) in input.iter().enumerate() {
            let l = x.to_u256().0;
            for j in 0..k {
                res[j * n + i] = l[j];
            }
        }
        for (j, rg) in self.rings.iter().enumerate() {
            t.copy_from_slice(&res[j * n..(j + 1) * n]);
            rg.inverse(&t, &mut res[j * n..(j + 1) * n]);
        }
        let mut r = [0u64; 4];
        for (i, o) in out.iter_mut().enumerate() {
            for j in 0..k {
                r[j] = res[j * n + i];
            }
            *o = W::from_u256(self.join(&r[..k]));
        }
        crate::rlwe::wipe_words(&mut res);
        crate::rlwe::wipe_words(&mut t);
    }
}

impl<W: Word> Ring<W> {
    /// The ring for a composite q = primes[0] ... primes[k-1] (distinct NTT
    /// primes below 2^62, 2 <= k <= BITS / 64) and the residues of the CRT
    /// root mod each prime; see [`Rns`]. q must be odd and below the word limit.
    pub fn new_rns(q: W, n: usize, primes: &[u64], roots: &[u64]) -> Result<Self, &'static str> {
        let k = primes.len();
        if k < 2 || k as u32 > W::BITS / 64 || roots.len() != k {
            return Err("an RNS modulus has 2 to BITS / 64 primes, one root each");
        }
        if n < 2 || !n.is_power_of_two() || n > (1 << 24) {
            return Err("n must be a power of two in [2, 2^24]");
        }
        if q < W::from_u128(3) || !q.bit0() || q.bits() > W::MAX_Q_BITS {
            return Err("unsupported q: need an odd q below the word limit");
        }
        let mut prod = U256::ONE;
        for (i, &p) in primes.iter().enumerate() {
            if p >> 62 != 0 || primes[..i].contains(&p) {
                return Err("the factors of q must be distinct primes below 2^62");
            }
            let (next, carry) = prod.mul_small_add(p, 0);
            if carry != 0 {
                return Err("q must be the product of the primes");
            }
            prod = next;
        }
        if prod != q.to_u256() {
            return Err("q must be the product of the primes");
        }
        let rings = primes.iter().zip(roots).map(|(&p, &rt)| Ring::<u64>::new(p, n, rt)).collect::<Result<Vec<_>, _>>()?;
        let limb = rings.iter().map(|rg| {
            let m = &rg.m;
            let c1 = m.rmod; // 2^64 mod p
            let c2 = m.mulp(c1, c1);
            let c3 = m.mulp(c2, c1);
            let one = m.reduce(1);
            [one, c1, c2, c3].map(|c| (c, m.shoup_quot(c)))
        }).collect();
        let mut inv = Vec::with_capacity(k);
        for (j, rg) in rings.iter().enumerate() {
            let m = &rg.m;
            let mut row = Vec::with_capacity(j);
            for &pi in &primes[..j] {
                let c = m.pow(m.reduce(pi), m.q - 2); // p_j is prime (Ring::new checked n^-1 the same way)
                if m.mulp(c, m.reduce(pi)) != 1 {
                    return Err("the factors of q must be coprime");
                }
                row.push((c, m.shoup_quot(c)));
            }
            inv.push(row);
        }
        Ok(Ring {
            m: Modulus::<W>::new(q),
            n,
            logn: n.trailing_zeros(),
            q,
            lazy: false,
            br: Vec::new(),
            twist: Vec::new(),
            twist_p: Vec::new(),
            scale: Vec::new(),
            scale_p: Vec::new(),
            fwd: Vec::new(),
            fwd_p: Vec::new(),
            inv: Vec::new(),
            inv_p: Vec::new(),
            rns: Some(Box::new(Rns { rings, limb, inv })),
        })
    }

    /// NTT-domain values as the NTT mod q in [0, q) (what ring_lwe_utils.ntt_forward
    /// returns): the CRT of the residues for an RNS ring, unchanged otherwise.
    pub fn ntt_to_std(&self, x: &mut [W]) {
        if let Some(r) = &self.rns {
            let k = r.rings.len();
            for v in x.iter_mut() {
                let l = v.to_u256().0;
                *v = W::from_u256(r.join(&l[..k]));
            }
        }
    }

    /// The inverse of ntt_to_std (the NTT mod q as NTT-domain values).
    pub fn ntt_from_std(&self, x: &mut [W]) {
        if let Some(r) = &self.rns {
            let k = r.rings.len();
            for v in x.iter_mut() {
                let l = v.to_u256().0;
                let mut limbs = [0u64; 4];
                for (j, o) in limbs.iter_mut().enumerate().take(k) {
                    *o = r.residue(j, &l);
                }
                *v = W::from_u256(U256(limbs));
            }
        }
    }
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
            lazy: q.bits() + 2 <= W::BITS,
            br: Vec::with_capacity(n),
            twist: Vec::with_capacity(n),
            twist_p: Vec::with_capacity(n),
            scale: Vec::with_capacity(n),
            scale_p: Vec::with_capacity(n),
            fwd: Vec::new(),
            fwd_p: Vec::new(),
            inv: Vec::new(),
            inv_p: Vec::new(),
            rns: None,
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
        if !self.lazy {
            return self.butterflies_reduced(x, tw, twp);
        }
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

    /// The butterflies with values in [0, q) throughout (inputs in [0, q)), for
    /// 2q < 2^BITS <= 4q.
    fn butterflies_reduced(&self, x: &mut [W], tw: &[W], twp: &[W]) {
        let q = self.m.q;
        let n = self.n;
        let mut off = 0;
        let mut len = 2;
        while len <= n {
            let half = len >> 1;
            let (t, tp) = (&tw[off..off + half], &twp[off..off + half]);
            for blk in x.chunks_exact_mut(len) {
                let (lo, hi) = blk.split_at_mut(half);
                for (((l, h), &w), &wp) in lo.iter_mut().zip(hi.iter_mut()).zip(t).zip(tp) {
                    let u = *l;
                    let v = self.m.shoup(*h, w, wp);
                    let v = if v >= q { v.wsub(q) } else { v };
                    let s = u.wadd(v);
                    *l = if s >= q { s.wsub(q) } else { s };
                    let d = u.wadd(q).wsub(v);
                    *h = if d >= q { d.wsub(q) } else { d };
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

    /// ring_lwe_utils.ntt_forward: twist, bit-reverse, stages; out in [0, q)
    /// (for an RNS ring: per-prime residues, see [`Rns`]).
    pub fn forward(&self, input: &[W], out: &mut [W]) {
        if let Some(r) = &self.rns {
            return r.forward(input, out, false);
        }
        for i in 0..self.n {
            let b = self.br[i] as usize;
            out[i] = self.m.shoup(input[b], self.twist[b], self.twist_p[b]);
        }
        if !self.lazy {
            let q = self.m.q;
            for v in out.iter_mut() {
                if *v >= q {
                    *v = v.wsub(q);
                }
            }
        }
        self.butterflies(out, &self.fwd, &self.fwd_p);
        for v in out.iter_mut() {
            *v = self.reduce4(*v);
        }
    }

    /// ring_lwe_utils.ntt_inverse: bit-reverse, stages, times n^-1 untwist[i]; out in [0, q).
    pub fn inverse(&self, input: &[W], out: &mut [W]) {
        if let Some(r) = &self.rns {
            return r.inverse(input, out);
        }
        for i in 0..self.n {
            out[i] = input[self.br[i] as usize];
        }
        if !self.lazy {
            let q = self.m.q;
            for v in out.iter_mut() {
                if *v >= q {
                    *v = v.wsub(q);
                }
            }
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
        if let Some(r) = &self.rns {
            return r.forward(input, out, true);
        }
        self.forward(input, out);
        for v in out.iter_mut() {
            *v = self.m.to_mont(*v);
        }
    }

    /// acc += a b pointwise in the NTT domain, a in the form of forward_mont.
    #[inline]
    pub fn pw_mul_add(&self, acc: &mut [W], a: &[W], b: &[W]) {
        if let Some(r) = &self.rns {
            for ((s, &x), &y) in acc.iter_mut().zip(a).zip(b) {
                let (mut l, xa, ya) = (s.to_u256().0, x.to_u256().0, y.to_u256().0);
                for (j, rg) in r.rings.iter().enumerate() {
                    l[j] = rg.m.add(l[j], rg.m.mul(xa[j], ya[j]));
                }
                *s = W::from_u256(U256(l));
            }
            return;
        }
        let m = &self.m;
        for ((s, &x), &y) in acc.iter_mut().zip(a).zip(b) {
            *s = m.add(*s, m.mul(x, y));
        }
    }

    /// out = a b pointwise in the NTT domain, a in the form of forward_mont.
    #[inline]
    pub fn pw_mul(&self, out: &mut [W], a: &[W], b: &[W]) {
        if let Some(r) = &self.rns {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                let (xa, ya) = (x.to_u256().0, y.to_u256().0);
                let mut l = [0u64; 4];
                for (j, rg) in r.rings.iter().enumerate() {
                    l[j] = rg.m.mul(xa[j], ya[j]);
                }
                *o = W::from_u256(U256(l));
            }
            return;
        }
        let m = &self.m;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = m.mul(x, y);
        }
    }

    /// The negacyclic product (for tests).
    pub fn poly_mul(&self, a: &[W], b: &[W]) -> Vec<W> {
        let (mut fa, mut fb, mut out) = (vec![W::ZERO; self.n], vec![W::ZERO; self.n], vec![W::ZERO; self.n]);
        if self.rns.is_some() {
            self.forward_mont(a, &mut fa);
            self.forward(b, &mut fb);
            let mut prod = vec![W::ZERO; self.n];
            self.pw_mul(&mut prod, &fa, &fb);
            self.inverse(&prod, &mut out);
            return out;
        }
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
    /// u64 words for q < 2^63, u128 words for q < 2^125, 256-bit words for q < 2^253.
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
    /// The ring of a composite q (Ring::new_rns): 128-bit words for two primes
    /// and q < 2^125, 256-bit words for up to four primes and q < 2^253.
    pub fn new_rns(q: U256, n: usize, primes: &[u64], roots: &[u64]) -> Result<Self, &'static str> {
        let bits = q.bits();
        if primes.len() <= 2 && bits <= u128::MAX_Q_BITS {
            Ok(AnyRing::W128(Ring::new_rns(u128::from_u256(q), n, primes, roots)?))
        } else if primes.len() <= 4 && bits <= U256::MAX_Q_BITS {
            Ok(AnyRing::W256(Ring::new_rns(q, n, primes, roots)?))
        } else {
            Err("an RNS modulus has at most four primes and q below 2^253")
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
pub(crate) mod tests {
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

    /// Distinct NTT primes of `bits` bits for dimension n (the first ones above 2^(bits - 1)).
    pub(crate) fn rns_primes(bits: u32, n: usize, k: usize) -> Vec<u64> {
        let mut out = Vec::new();
        let mut c = ntt_prime(bits, n);
        while out.len() < k {
            out.push(c.0[0]);
            c = c.wadd(U256::from_u128(2 * n as u128));
            while !is_prime(c) {
                c = c.wadd(U256::from_u128(2 * n as u128));
            }
        }
        out
    }

    pub(crate) fn rns_ring(bits: u32, n: usize, k: usize) -> (AnyRing, U256) {
        let ps = rns_primes(bits, n, k);
        let roots: Vec<u64> = ps.iter().map(|&p| primitive_root(p, n)).collect();
        let q = ps.iter().fold(U256::ONE, |acc, &p| acc.mul_small_add(p, 0).0);
        (AnyRing::new_rns(q, n, &ps, &roots).unwrap(), q)
    }

    #[test]
    fn rns_products_match_schoolbook() {
        for &(bits, n, k) in &[(30u32, 8usize, 2usize), (50, 16, 2), (40, 16, 3), (54, 32, 4), (61, 16, 4)] {
            let (ring, q) = rns_ring(bits, n, k);
            assert_eq!(ring.word_bits(), if k <= 2 && q.bits() <= 125 { 128 } else { 256 });
            let mut rng = Lcg(q.0[0] as u128 ^ 0x5eed);
            let a: Vec<U256> = (0..n).map(|_| rng.below(q)).collect();
            let mut b: Vec<U256> = (0..n).map(|_| rng.below(q)).collect();
            b[0] = q.wsub(U256::ONE);
            let want = schoolbook(&a, &b, q);
            with_ring!(&ring, r => {
                let aw: Vec<_> = a.iter().map(|&v| Word::from_u256(v)).collect();
                let bw: Vec<_> = b.iter().map(|&v| Word::from_u256(v)).collect();
                let got: Vec<U256> = r.poly_mul(&aw, &bw).iter().map(|v| v.to_u256()).collect();
                assert_eq!(got, want, "bits {} n {} k {}", bits, n, k);
                let mut f = aw.clone();
                let mut back = aw.clone();
                r.forward(&aw, &mut f);
                r.inverse(&f, &mut back);
                assert_eq!(back, aw);
                // the NTT-domain values read as the NTT mod q and back
                let mut std = f.clone();
                r.ntt_to_std(&mut std);
                assert!(std.iter().all(|v| v.to_u256() < q));
                r.ntt_from_std(&mut std);
                assert_eq!(std, f);
                // sums of products in the NTT domain
                let (mut fa, mut fb, mut acc) = (aw.clone(), bw.clone(), vec![Word::ZERO; n]);
                r.forward_mont(&aw, &mut fa);
                r.forward(&bw, &mut fb);
                r.pw_mul_add(&mut acc, &fa, &fb);
                r.pw_mul_add(&mut acc, &fa, &fb);
                let mut two = aw.clone();
                r.inverse(&acc, &mut two);
                let m = Modulus::<U256>::new(q);
                assert_eq!(two.iter().map(|v| v.to_u256()).collect::<Vec<_>>(),
                           want.iter().map(|&v| m.add(v, v)).collect::<Vec<_>>());
            });
        }
        // refused: a wrong product, repeated or too large factors, too many factors
        let ps = rns_primes(50, 16, 3);
        let roots: Vec<u64> = ps.iter().map(|&p| primitive_root(p, 16)).collect();
        let q = ps.iter().fold(U256::ONE, |acc, &p| acc.mul_small_add(p, 0).0);
        assert!(AnyRing::new_rns(q.wadd(U256::from_u128(2)), 16, &ps, &roots).is_err());
        assert!(AnyRing::new_rns(q, 16, &[ps[0], ps[0], ps[1]], &roots).is_err());
        assert!(AnyRing::new_rns(q, 16, &ps[..1], &roots[..1]).is_err());
        assert!(Ring::<u128>::new_rns(q.to_u128(), 16, &ps, &roots).is_err()); // three limbs in a u128
        let big = rns_primes(63, 16, 2);
        let qb = big.iter().fold(U256::ONE, |acc, &p| acc.mul_small_add(p, 0).0);
        let rb: Vec<u64> = big.iter().map(|&p| primitive_root(p, 16)).collect();
        assert!(AnyRing::new_rns(qb, 16, &big, &rb).is_err());
    }
}
