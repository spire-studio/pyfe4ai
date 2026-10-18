//! BMMS21 MIFE / MCFE kernels for an RNS modulus q = q1 q2 (two NTT primes
//! below 2^62): polynomials are kept as residues mod q1 and q2 (one u64 NTT
//! per prime) and converted to and from the standard representation in
//! [0, q) where they cross the boundary (Garner). Every kernel computes the
//! values of the Python code with the composite q and the root that is the
//! CRT of the primes' roots (ring_lwe_utils._get_ntt_root), as the kernels of
//! `rlwe` do for a prime q.

use crate::arith::Modulus;
use crate::facct::{self, RandBits};
use crate::gauss::{gauss_bytes, gauss_fill};
use crate::keccak::{hash_to_zq_mode, HashMode};
use crate::ntt::Ring;
use crate::parallel_for;
use crate::rlwe::{
    chunk_bytes, decode_scaled, facct_chunks, tensor_chunk_bytes, wipe_bytes, wipe_words, Error, FacctNoise, Noise,
    Randomness,
};

/// Z_q[x] / (x^n + 1) for q = q1 q2. An RNS polynomial is 2n words: the n
/// residues mod q1, then the n residues mod q2.
#[derive(Clone, Debug)]
pub struct RnsRing {
    pub n: usize,
    pub logn: u32,
    pub q: u128,
    pub qbits: u32,
    pub r: [Ring<u64>; 2],
    /// Montgomery arithmetic mod the composite q (values in [0, q)).
    pub m: Modulus<u128>,
    q1: u64,
    /// q1^-1 mod q2 and its Shoup quotient.
    q1inv: u64,
    q1inv_p: u64,
}

impl RnsRing {
    /// The ring for q = primes[0] primes[1] (distinct NTT primes below 2^62) and
    /// primitive 2n-th roots of unity mod each prime.
    pub fn new(q: u128, n: usize, primes: [u64; 2], roots: [u64; 2]) -> Result<Self, &'static str> {
        if primes[0] == primes[1] || (primes[0] as u128).checked_mul(primes[1] as u128) != Some(q) {
            return Err("q must be the product of two distinct primes");
        }
        if q >> 124 != 0 {
            return Err("q must be below 2^124");
        }
        let r = [Ring::new(primes[0], n, roots[0])?, Ring::new(primes[1], n, roots[1])?];
        let m2 = &r[1].m;
        // q1^-1 mod q2 by Fermat (q2 is prime; Ring::new checked n^-1 the same way)
        let q1inv = m2.pow(m2.reduce(primes[0]), primes[1] - 2);
        if m2.mulp(m2.reduce(primes[0]), q1inv) != 1 {
            return Err("the factors of q must be coprime");
        }
        let q1inv_p = m2.shoup_quot(q1inv);
        Ok(RnsRing { n, logn: n.trailing_zeros(), q, qbits: 128 - q.leading_zeros(), m: Modulus::new(q), q1: primes[0],
                     r, q1inv, q1inv_p })
    }

    /// x mod q_i for any 128-bit x.
    #[inline(always)]
    fn res(m: &Modulus<u64>, x: u128) -> u64 {
        m.add(m.to_mont((x >> 64) as u64), m.reduce(x as u64))
    }

    /// The residues of standard values (n of them) into `out` (2n).
    pub fn split(&self, x: &[u128], out: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            for (o, &v) in out[i * n..(i + 1) * n].iter_mut().zip(x) {
                *o = Self::res(&rg.m, v);
            }
        }
    }

    /// The value in [0, q) of residues (x1, x2) (Garner): x1 + q1 ((x2 - x1) q1^-1 mod q2).
    #[inline(always)]
    pub fn join1(&self, x1: u64, x2: u64) -> u128 {
        let m2 = &self.r[1].m;
        let t = m2.shoup(m2.sub(x2, m2.reduce(x1)), self.q1inv, self.q1inv_p);
        let t = if t >= m2.q { t - m2.q } else { t };
        x1 as u128 + self.q1 as u128 * t as u128
    }

    /// Standard values of an RNS polynomial (2n residues) into `out` (n).
    pub fn join(&self, x: &[u64], out: &mut [u128]) {
        let n = self.n;
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.join1(x[i], x[n + i]);
        }
    }

    /// Small signed integers (samples, plaintexts) as residues.
    pub fn small(&self, v: &[i64], out: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            for (o, &s) in out[i * n..(i + 1) * n].iter_mut().zip(v) {
                *o = rg.m.reduce_i128(s as i128);
            }
        }
    }

    /// The forward NTT of each prime (the Python NTT mod q, reduced mod q_i).
    pub fn forward(&self, x: &[u64], out: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            rg.forward(&x[i * n..(i + 1) * n], &mut out[i * n..(i + 1) * n]);
        }
    }

    pub fn inverse(&self, x: &[u64], out: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            rg.inverse(&x[i * n..(i + 1) * n], &mut out[i * n..(i + 1) * n]);
        }
    }

    /// Forward NTT in Montgomery form per prime (the key format).
    pub fn forward_mont(&self, x: &[u64], out: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            rg.forward_mont(&x[i * n..(i + 1) * n], &mut out[i * n..(i + 1) * n]);
        }
    }

    /// NTT(x) in Montgomery form of standard values x (n), as 2n residues.
    pub fn ntt_mont_std(&self, x: &[u128]) -> Vec<u64> {
        let mut r = vec![0u64; 2 * self.n];
        let mut out = vec![0u64; 2 * self.n];
        self.split(x, &mut r);
        self.forward_mont(&r, &mut out);
        out
    }

    /// out = a * b pointwise (a in Montgomery form), per prime.
    #[inline]
    fn mul(&self, a: &[u64], b: &[u64], out: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            let m = &rg.m;
            for j in i * n..(i + 1) * n {
                out[j] = m.mul(a[j], b[j]);
            }
        }
    }

    /// acc += a * b pointwise (a in Montgomery form), per prime.
    #[inline]
    fn mul_add(&self, a: &[u64], b: &[u64], acc: &mut [u64]) {
        let n = self.n;
        for (i, rg) in self.r.iter().enumerate() {
            let m = &rg.m;
            for j in i * n..(i + 1) * n {
                acc[j] = m.add(acc[j], m.mul(a[j], b[j]));
            }
        }
    }

    /// The negacyclic product of standard values (for tests).
    pub fn poly_mul(&self, a: &[u128], b: &[u128]) -> Vec<u128> {
        let n2 = 2 * self.n;
        let (mut fa, mut fb, mut t, mut o) = (vec![0u64; n2], vec![0u64; n2], vec![0u64; n2], vec![0u64; n2]);
        self.split(a, &mut t);
        self.forward(&t, &mut fa);
        self.split(b, &mut t);
        self.forward_mont(&t, &mut fb);
        self.mul(&fb, &fa, &mut t);
        self.inverse(&t, &mut o);
        let mut out = vec![0u128; self.n];
        self.join(&o, &mut out);
        out
    }
}

/// Key of one slot (rlwe::EncKey for an RNS modulus): NTT(a), NTT(pk_row) in
/// Montgomery form and the pads u_row, as residues; Delta; how label pads are
/// hashed (MCFE; HashMode::Wide256 unless set).
#[derive(Clone, Debug)]
pub struct RnsKey {
    pub eta: usize,
    pub a_hat: Vec<u64>,
    pub pk_hat: Vec<u64>,
    pub u: Option<Vec<u64>>,
    pub delta: u128,
    /// Delta mod q_i in Montgomery form, and mod q.
    pub delta_m: [u64; 2],
    pub delta_mq: u128,
    pub hash_mode: HashMode,
}

impl Drop for RnsKey {
    /// The pads u are secret: wiped when the key goes.
    fn drop(&mut self) {
        if let Some(u) = self.u.as_mut() {
            wipe_words(u);
        }
    }
}

impl RnsKey {
    /// From a, the eta pk rows and the pad rows (standard values, n each) and Delta.
    pub fn new(ring: &RnsRing, a: &[u128], pk: &[Vec<u128>], u: Option<&[Vec<u128>]>, delta: u128) -> Self {
        let n2 = 2 * ring.n;
        let mut pk_hat = Vec::with_capacity(n2 * pk.len());
        for row in pk {
            pk_hat.extend(ring.ntt_mont_std(row));
        }
        let u = u.map(|rows| {
            let mut out = vec![0u64; n2 * rows.len()];
            for (row, dst) in rows.iter().zip(out.chunks_exact_mut(n2)) {
                ring.split(row, dst);
            }
            out
        });
        let dm = |i: usize| ring.r[i].m.to_mont(RnsRing::res(&ring.r[i].m, delta));
        RnsKey {
            eta: pk.len(),
            a_hat: ring.ntt_mont_std(a),
            pk_hat,
            u,
            delta,
            delta_m: [dm(0), dm(1)],
            delta_mq: ring.m.to_mont(ring.m.reduce(delta)),
            hash_mode: HashMode::Wide256,
        }
    }
}

/// rlwe::encrypt_chunk_with for an RNS modulus: w as residues (eta rows of
/// 2n), ct0 (eta rows of n) and ct1 (n) as standard values; `draw(wide, out)`
/// samples one vector (wide: sigma3), called for r, the e_row, e in that order,
/// as for a prime modulus.
fn encrypt_chunk_with(
    ring: &RnsRing, key: &RnsKey, w: &[u64], mut draw: impl FnMut(bool, &mut [i64]), ct0: &mut [u128],
    ct1: &mut [u128],
) {
    let n = ring.n;
    let n2 = 2 * n;
    let mut smp = vec![0i64; n];
    let (mut r, mut rh, mut t, mut o) = (vec![0u64; n2], vec![0u64; n2], vec![0u64; n2], vec![0u64; n2]);
    draw(false, &mut smp);
    ring.small(&smp, &mut r);
    ring.forward(&r, &mut rh);
    for row in 0..key.eta {
        ring.mul(&key.pk_hat[row * n2..(row + 1) * n2], &rh, &mut t);
        ring.inverse(&t, &mut o);
        draw(true, &mut smp);
        let wr = &w[row * n2..(row + 1) * n2];
        for (i, rg) in ring.r.iter().enumerate() {
            let m = &rg.m;
            for j in 0..n {
                let k = i * n + j;
                o[k] = m.add(m.add(o[k], m.reduce_i128(smp[j] as i128)), m.mul(wr[k], key.delta_m[i]));
            }
        }
        ring.join(&o, &mut ct0[row * n..(row + 1) * n]);
    }
    ring.mul(&key.a_hat, &rh, &mut t);
    ring.inverse(&t, &mut o);
    draw(false, &mut smp);
    for (i, rg) in ring.r.iter().enumerate() {
        let m = &rg.m;
        for j in 0..n {
            o[i * n + j] = m.add(o[i * n + j], m.reduce_i128(smp[j] as i128));
        }
    }
    ring.join(&o, ct1);
    for v in smp.iter_mut() {
        unsafe { std::ptr::write_volatile(v, 0) };
    }
    // r, NTT(r) and the products with it determine the noise-free ciphertext
    wipe_words(&mut r);
    wipe_words(&mut rh);
    wipe_words(&mut t);
    wipe_words(&mut o);
}

/// encrypt_chunk_with for the rounded Gaussian: gauss_bytes(n) random bytes per
/// vector, consumed in order from `rnd` (rlwe::encrypt_chunk).
#[allow(clippy::too_many_arguments)]
pub fn encrypt_chunk(
    ring: &RnsRing, key: &RnsKey, w: &[u64], sigma2: f64, sigma3: f64, rnd: &[u8], ct0: &mut [u128], ct1: &mut [u128],
) {
    let per = gauss_bytes(ring.n);
    let mut pos = 0;
    encrypt_chunk_with(ring, key, w, |wide, out| {
        gauss_fill(&rnd[pos..], if wide { sigma3 } else { sigma2 }, out);
        pos += per;
    }, ct0, ct1);
}

/// encrypt_chunk_with for FACCT, drawing getrandbits from `bits` as the Python
/// sampler draws them from its source (rlwe::encrypt_chunk_facct).
pub fn encrypt_chunk_facct(
    ring: &RnsRing, key: &RnsKey, w: &[u64], noise: &FacctNoise, bits: &mut dyn RandBits, ct0: &mut [u128],
    ct1: &mut [u128],
) {
    encrypt_chunk_with(ring, key, w, |wide, out| {
        if wide {
            facct::sample(noise.sigma3, out, bits, noise.exact3)
        } else {
            facct::sample(noise.sigma2, out, bits, noise.exact2)
        }
    }, ct0, ct1);
}

/// rlwe's pad_row: H_c = hash_to_zq_vector(material, mode) plus u, as
/// residues. The values are computed mod q in 128-bit words, then split
/// (faster than one reduction per prime: two 64-bit limbs instead of four per
/// prime).
fn pad_row(ring: &RnsRing, u: &[u64], material: Option<&[u8]>, mode: HashMode, out: &mut [u64]) {
    let n = ring.n;
    match material {
        Some(mat) => {
            let mut h = vec![0u128; n];
            hash_to_zq_mode(&ring.m, mat, mode, &mut h);
            ring.split(&h, out);
            wipe_words(&mut h); // u' is secret: so is H_c
        }
        None => out.iter_mut().for_each(|v| *v = 0),
    }
    for (i, rg) in ring.r.iter().enumerate() {
        for j in i * n..(i + 1) * n {
            out[j] = rg.m.add(u[j], out[j]);
        }
    }
}

/// w = u + H_c + x (zero-extended) of chunk j as residues.
fn chunk_w(ring: &RnsRing, key: &RnsKey, u: &[u64], material: Option<&[u8]>, x: &[i64], j: usize, w: &mut [u64]) {
    let n = ring.n;
    pad_row(ring, u, material, key.hash_mode, w);
    let k0 = j * n;
    let kk = n.min(x.len().saturating_sub(k0));
    for (i, rg) in ring.r.iter().enumerate() {
        for t in 0..kk {
            w[i * n + t] = rg.m.add(w[i * n + t], rg.m.reduce_i128(x[k0 + t] as i128));
        }
    }
}

/// rlwe::encrypt_tensor for an RNS modulus (eta = 1), with the rounded or
/// the FACCT noise; `emit(c, ct0, ct1)` with standard values.
#[allow(clippy::too_many_arguments)]
pub fn encrypt_tensor<F>(
    ring: &RnsRing, key: &RnsKey, x: &[i64], materials: Option<&[&[u8]]>, sigma2: f64, sigma3: f64, noise: &Noise,
    rnd: Randomness, threads: usize, emit: F,
) -> Result<usize, Error>
where
    F: Fn(usize, &[u128], &[u128]) + Sync,
{
    let n = ring.n;
    let u = key.u.as_ref().ok_or(Error::Input("encrypt_tensor needs the pad u"))?;
    if key.eta != 1 || x.is_empty() {
        return Err(Error::Input("encrypt_tensor needs eta = 1 and a non-empty layer"));
    }
    let chunks = (x.len() + n - 1) / n;
    if materials.map_or(false, |m| m.len() != chunks) {
        return Err(Error::Input("one pad material per chunk"));
    }
    let plain = |j: usize| {
        let mut w = vec![0u64; 2 * n];
        chunk_w(ring, key, u, materials.map(|m| m[j]), x, j, &mut w);
        w
    };
    if let Noise::Facct(nz) = noise {
        facct_chunks(chunks, threads, &rnd, |j, bits| {
            let (mut c0, mut c1) = (vec![0u128; n], vec![0u128; n]);
            let mut w = plain(j);
            encrypt_chunk_facct(ring, key, &w, nz, bits, &mut c0, &mut c1);
            wipe_words(&mut w); // x + u + H_c: the pad u is secret
            emit(j, &c0, &c1);
        })?;
        return Ok(chunks);
    }
    let per = tensor_chunk_bytes(n);
    if let Randomness::Given(b) = rnd {
        if b.len() < per * chunks {
            return Err(Error::Input("not enough random bytes"));
        }
    }
    let failed = std::sync::Mutex::new(None);
    parallel_for(chunks, threads, |j| {
        let mut own = Vec::new();
        let (mut c0, mut c1) = (vec![0u128; n], vec![0u128; n]);
        match chunk_bytes(&rnd, j, per, &mut own) {
            Ok(rb) => {
                let mut w = plain(j);
                encrypt_chunk(ring, key, &w, sigma2, sigma3, rb, &mut c0, &mut c1);
                wipe_words(&mut w);
            }
            Err(e) => {
                *failed.lock().unwrap() = Some(e);
                return;
            }
        }
        wipe_bytes(&mut own);
        emit(j, &c0, &c1);
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(Error::Random(e));
    }
    Ok(chunks)
}

/// rlwe::decrypt_chunk for an RNS modulus: z, the slots ((eta + 1) n values
/// each) and `ym` (weights in Montgomery form mod q) are standard values,
/// `sk_hats[s]` NTT(sk_s) per prime in Montgomery form (2n residues).
#[allow(clippy::too_many_arguments)]
pub fn decrypt_chunk(
    ring: &RnsRing, eta: usize, z: &[u128], slots: &[&[u128]], ym: &[u128], sk_hats: &[&[u64]], delta: u128,
    bound: i128, out: &mut [i64], none: &mut [u8],
) -> bool {
    let n = ring.n;
    let n2 = 2 * n;
    let m = &ring.m;
    let delta_m = m.to_mont(delta);
    let mut tot = vec![0u128; n];
    let (mut acc, mut x, mut xh, mut o) = (vec![0u64; n2], vec![0u64; n2], vec![0u64; n2], vec![0u64; n2]);
    for i in 0..n {
        tot[i] = m.sub(0, m.mul(z[i], delta_m));
    }
    for (s, slot) in slots.iter().enumerate() {
        for r in 0..eta {
            let y = ym[s * eta + r];
            let row = &slot[r * n..(r + 1) * n];
            for i in 0..n {
                tot[i] = m.add(tot[i], m.mul(row[i], y));
            }
        }
        ring.split(&slot[eta * n..(eta + 1) * n], &mut x);
        ring.forward(&x, &mut xh);
        ring.mul_add(sk_hats[s], &xh, &mut acc);
    }
    ring.inverse(&acc, &mut o);
    for i in 0..out.len() {
        let v = m.sub(tot[i], ring.join1(o[i], o[n + i]));
        match decode_scaled(v, ring.q, delta, bound) {
            Ok(Some(mu)) => {
                out[i] = mu;
                none[i] = 0;
            }
            Ok(None) => {
                out[i] = 0;
                none[i] = 1;
            }
            Err(()) => return false,
        }
    }
    true
}

/// rlwe::pad_terms for an RNS modulus (standard values, arithmetic mod q), with `mode`.
#[allow(clippy::too_many_arguments)]
pub fn pad_terms<F>(
    ring: &RnsRing, base: &[u128], materials: &[Vec<&[u8]>], ym: &[u128], eta: usize, mode: HashMode, threads: usize,
    emit: F,
) where
    F: Fn(usize, &[u128]) + Sync,
{
    let n = ring.n;
    let m = &ring.m;
    parallel_for(materials.len(), threads, |l| {
        let mut z = base.to_vec();
        let mut h = vec![0u128; n * eta];
        for (s, mat) in materials[l].iter().enumerate() {
            hash_to_zq_mode(m, mat, mode, &mut h);
            for r in 0..eta {
                let y = ym[s * eta + r];
                for i in 0..n {
                    z[i] = m.add(z[i], m.mul(h[r * n + i], y));
                }
            }
        }
        wipe_words(&mut h); // the pads of a client's u'
        emit(l, &z);
        wipe_words(&mut z);
    });
}

/// rlwe::Precomputed for an RNS modulus: per chunk P = pk r + e_0 + Delta (u + H_c)
/// and ct1, standard values; one-time, wiped when used and on drop.
pub struct RnsPrecomputed {
    pub n: usize,
    pub data: Vec<u128>,
    pub used: Vec<bool>,
}

impl RnsPrecomputed {
    fn wipe(&mut self, j: usize) {
        let n = self.n;
        for v in self.data[2 * n * j..2 * n * (j + 1)].iter_mut() {
            unsafe { std::ptr::write_volatile(v, 0) };
        }
    }
    pub fn chunks(&self) -> usize {
        self.used.len()
    }
}

impl Drop for RnsPrecomputed {
    fn drop(&mut self) {
        for j in 0..self.used.len() {
            self.wipe(j);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn precompute(
    ring: &RnsRing, key: &RnsKey, materials: &[&[u8]], sigma2: f64, sigma3: f64, noise: &Noise, rnd: Randomness,
    threads: usize,
) -> Result<RnsPrecomputed, Error> {
    let n = ring.n;
    let u = key.u.as_ref().ok_or(Error::Input("precompute needs the pad u"))?;
    if key.eta != 1 {
        return Err(Error::Input("precompute needs eta = 1"));
    }
    let chunks = materials.len();
    let data: Vec<std::sync::Mutex<Vec<u128>>> = (0..chunks).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let pad = |j: usize| {
        let mut w = vec![0u64; 2 * n];
        pad_row(ring, u, Some(materials[j]), key.hash_mode, &mut w);
        w
    };
    let collect = |data: Vec<std::sync::Mutex<Vec<u128>>>| {
        let mut flat = Vec::with_capacity(2 * n * chunks); // no reallocation: nothing left behind
        for d in data {
            let mut pc = d.into_inner().unwrap();
            flat.extend_from_slice(&pc);
            wipe_words(&mut pc);
        }
        RnsPrecomputed { n, data: flat, used: vec![false; chunks] }
    };
    if let Noise::Facct(nz) = noise {
        facct_chunks(chunks, threads, &rnd, |j, bits| {
            let mut pc = vec![0u128; 2 * n];
            let (c0, c1) = pc.split_at_mut(n);
            let mut w = pad(j);
            encrypt_chunk_facct(ring, key, &w, nz, bits, c0, c1);
            wipe_words(&mut w);
            *data[j].lock().unwrap() = pc;
        })?;
        return Ok(collect(data));
    }
    let per = tensor_chunk_bytes(n);
    if let Randomness::Given(b) = rnd {
        if b.len() < per * chunks {
            return Err(Error::Input("not enough random bytes"));
        }
    }
    let failed = std::sync::Mutex::new(None);
    parallel_for(chunks, threads, |j| {
        let mut own = Vec::new();
        let mut pc = vec![0u128; 2 * n];
        match chunk_bytes(&rnd, j, per, &mut own) {
            Ok(rb) => {
                let (c0, c1) = pc.split_at_mut(n);
                let mut w = pad(j);
                encrypt_chunk(ring, key, &w, sigma2, sigma3, rb, c0, c1);
                wipe_words(&mut w);
            }
            Err(e) => {
                *failed.lock().unwrap() = Some(e);
                return;
            }
        }
        wipe_bytes(&mut own); // the samples' bytes (empty when given)
        *data[j].lock().unwrap() = pc;
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(Error::Random(e));
    }
    Ok(collect(data))
}

/// rlwe::encrypt_precomputed for an RNS modulus: ct0 = P + Delta x mod q.
pub fn encrypt_precomputed<F>(
    ring: &RnsRing, key: &RnsKey, pre: &mut RnsPrecomputed, first: usize, x: &[i64], threads: usize, emit: F,
) -> Result<usize, Error>
where
    F: Fn(usize, &[u128], &[u128]) + Sync,
{
    let n = ring.n;
    if x.is_empty() {
        return Err(Error::Input("x must be non-empty"));
    }
    let chunks = (x.len() + n - 1) / n;
    if first.checked_add(chunks).map_or(true, |end| end > pre.chunks()) {
        return Err(Error::Input("the precomputation does not cover these chunks"));
    }
    if pre.used[first..first + chunks].iter().any(|&u| u) {
        return Err(Error::Input("precomputed chunk already used"));
    }
    for u in pre.used[first..first + chunks].iter_mut() {
        *u = true;
    }
    {
        let data = &pre.data;
        let m = &ring.m;
        parallel_for(chunks, threads, |j| {
            let jj = first + j;
            let mut c0 = data[2 * n * jj..2 * n * jj + n].to_vec();
            let c1 = &data[2 * n * jj + n..2 * n * (jj + 1)];
            let k0 = j * n;
            for i in 0..n.min(x.len() - k0) {
                c0[i] = m.add(c0[i], m.mul(m.reduce_i128(x[k0 + i] as i128), key.delta_mq));
            }
            emit(j, &c0, c1);
            for v in c0.iter_mut() {
                unsafe { std::ptr::write_volatile(v, 0) };
            }
        });
    }
    for j in first..first + chunks {
        pre.wipe(j);
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::tests::Lcg;
    use crate::arith::{Word, U256};

    fn root_of(q: u64, n: usize) -> u64 {
        let m = Modulus::<u64>::new(q);
        let cof = (q - 1) / (2 * n as u64);
        (2u64..1000).map(|g| m.pow(g, cof)).find(|&r| m.pow(r, n as u64) == q - 1).unwrap()
    }

    fn is_prime(p: u64) -> bool {
        if p < 4 {
            return p > 1;
        }
        let m = Modulus::<u64>::new(p);
        let (mut d, mut s) = (p - 1, 0);
        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }
        'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            let mut x = m.pow(a % p, d);
            if x <= 1 || x == p - 1 {
                continue;
            }
            for _ in 1..s {
                x = m.mulp(x, x);
                if x == p - 1 {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    /// The first prime = 1 mod 2n above 2^(bits - 1).
    fn ntt_prime_u64(bits: u32, n: usize) -> u64 {
        let step = 2 * n as u64;
        let mut c = (1u64 << (bits - 1)) / step * step + 1;
        while c < 1u64 << (bits - 1) || !is_prime(c) {
            c += step;
        }
        c
    }

    fn ring(bits: u32, n: usize) -> RnsRing {
        let q1 = ntt_prime_u64(bits, n);
        let q2 = ntt_prime_u64(bits + 1, n);
        RnsRing::new(q1 as u128 * q2 as u128, n, [q1, q2], [root_of(q1, n), root_of(q2, n)]).unwrap()
    }

    fn schoolbook(a: &[u128], b: &[u128], q: u128) -> Vec<u128> {
        let n = a.len();
        let m = Modulus::<U256>::new(U256::from_u128(q));
        let mut out = vec![U256::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                let p = crate::arith::tests::slow_mulmod(U256::from_u128(a[i]), U256::from_u128(b[j]), m.q);
                if i + j < n {
                    out[i + j] = m.add(out[i + j], p);
                } else {
                    out[i + j - n] = m.sub(out[i + j - n], p);
                }
            }
        }
        out.iter().map(|v| v.to_u128()).collect()
    }

    #[test]
    fn crt_round_trip_and_products() {
        for &(bits, n) in &[(20u32, 8usize), (39, 16), (40, 64), (61, 32)] {
            let r = ring(bits, n);
            let mut rng = Lcg(r.q ^ 0x1234_5678);
            let q = U256::from_u128(r.q);
            let a: Vec<u128> = (0..n).map(|_| rng.below(q).to_u128()).collect();
            let b: Vec<u128> = (0..n).map(|_| rng.below(q).to_u128()).collect();
            let mut res = vec![0u64; 2 * n];
            let mut back = vec![0u128; n];
            r.split(&a, &mut res);
            r.join(&res, &mut back);
            assert_eq!(back, a);
            for &v in &[0u128, 1, r.q - 1, (1u128 << 64) % r.q, r.q / 2] {
                let (x1, x2) = (RnsRing::res(&r.r[0].m, v), RnsRing::res(&r.r[1].m, v));
                assert_eq!(r.join1(x1, x2), v);
            }
            assert_eq!(r.poly_mul(&a, &b), schoolbook(&a, &b, r.q), "bits {} n {}", bits, n);
        }
        let r = ring(39, 16);
        assert!(RnsRing::new(r.q + 2, 16, [r.r[0].q, r.r[1].q], [root_of(r.r[0].q, 16), root_of(r.r[1].q, 16)]).is_err());
    }
}
