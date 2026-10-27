//! BMMS21 MIFE / MCFE over Ring-LWE: the loops of pyfe4ai.schemes.mife.ring_lwe
//! and mcfe.ring_lwe on words, for one chunk or a whole tensor.

use crate::arith::{Word, U256};
use crate::facct::{self, ByteBits, RandBits};
use crate::gauss::{gauss_bytes, gauss_fill};
use crate::keccak::{hash_to_zq_mode, HashMode};
use crate::ntt::Ring;
use crate::{osrand, parallel_for};

/// Key of one slot: NTT(a) and NTT(pk_row) in Montgomery form, the pads u_row, Delta,
/// and how label pads are hashed (MCFE; HashMode::Wide256 unless set).
#[derive(Clone, Debug)]
pub struct EncKey<W: Word> {
    pub eta: usize,
    pub a_hat: Vec<W>,
    pub pk_hat: Vec<W>,
    pub u: Option<Vec<W>>,
    pub delta: W,
    pub delta_m: W,
    pub hash_mode: HashMode,
}

impl<W: Word> Drop for EncKey<W> {
    /// The pads u are secret: wiped when the key goes.
    fn drop(&mut self) {
        if let Some(u) = self.u.as_mut() {
            wipe_words(u);
        }
    }
}

impl<W: Word> EncKey<W> {
    /// From a, the eta pk rows and the pad rows (plain words mod q, n each) and Delta.
    pub fn new(ring: &Ring<W>, a: &[W], pk: &[Vec<W>], u: Option<&[Vec<W>]>, delta: W) -> Self {
        let n = ring.n;
        let mut a_hat = vec![W::ZERO; n];
        ring.forward_mont(a, &mut a_hat);
        let mut pk_hat = vec![W::ZERO; n * pk.len()];
        for (row, dst) in pk.iter().zip(pk_hat.chunks_exact_mut(n)) {
            ring.forward_mont(row, dst);
        }
        EncKey {
            eta: pk.len(),
            a_hat,
            pk_hat,
            u: u.map(|rows| rows.concat()),
            delta,
            delta_m: ring.m.to_mont(delta),
            hash_mode: HashMode::Wide256,
        }
    }
}

/// One chunk of BMMS21 encryption (MIFERingLWE._encrypt_rows): r <- D(sigma2);
/// ct0_row = pk_row r + e_row + Delta w_row, e_row <- D(sigma3); ct1 = a r + e,
/// e <- D(sigma2). `draw(wide, out)` samples one vector (wide: sigma3), called
/// for r, the e_row, e in that order.
fn encrypt_chunk_with<W: Word>(
    ring: &Ring<W>, key: &EncKey<W>, w: &[W], mut draw: impl FnMut(bool, &mut [i64]), ct0: &mut [W],
    ct1: &mut [W],
) {
    let n = ring.n;
    let m = &ring.m;
    let mut smp = vec![0i64; n];
    let (mut r, mut rh, mut t) = (vec![W::ZERO; n], vec![W::ZERO; n], vec![W::ZERO; n]);
    draw(false, &mut smp);
    for i in 0..n {
        r[i] = m.reduce_i128(smp[i] as i128);
    }
    ring.forward(&r, &mut rh);
    for row in 0..key.eta {
        ring.pw_mul(&mut t, &key.pk_hat[row * n..(row + 1) * n], &rh);
        let out = &mut ct0[row * n..(row + 1) * n];
        ring.inverse(&t, out);
        draw(true, &mut smp);
        let wr = &w[row * n..(row + 1) * n];
        for i in 0..n {
            out[i] = m.add(m.add(out[i], m.reduce_i128(smp[i] as i128)), m.mul(wr[i], key.delta_m));
        }
    }
    ring.pw_mul(&mut t, &key.a_hat, &rh);
    ring.inverse(&t, ct1);
    draw(false, &mut smp);
    for i in 0..n {
        ct1[i] = m.add(ct1[i], m.reduce_i128(smp[i] as i128));
    }
    for v in smp.iter_mut() {
        unsafe { std::ptr::write_volatile(v, 0) };
    }
    // r, NTT(r) and the products with it determine the noise-free ciphertext
    wipe_words(&mut r);
    wipe_words(&mut rh);
    wipe_words(&mut t);
}

/// encrypt_chunk_with for the rounded Gaussian of discrete_gaussian_matrix:
/// gauss_bytes(n) random bytes per vector, consumed in order from `rnd`.
pub fn encrypt_chunk<W: Word>(
    ring: &Ring<W>, key: &EncKey<W>, w: &[W], sigma2: f64, sigma3: f64, rnd: &[u8],
    ct0: &mut [W], ct1: &mut [W],
) {
    let per = gauss_bytes(ring.n);
    let mut pos = 0;
    encrypt_chunk_with(ring, key, w, |wide, out| {
        gauss_fill(&rnd[pos..], if wide { sigma3 } else { sigma2 }, out);
        pos += per;
    }, ct0, ct1);
}

/// The exact sampler (discrete_gaussian_facct) for both widths: the tables
/// and the exact acceptance thresholds of the Python reference.
pub struct FacctNoise<'a> {
    pub sigma2: &'a facct::Table,
    pub sigma3: &'a facct::Table,
    pub exact2: &'a (dyn Fn(u64, u64) -> U256 + Sync),
    pub exact3: &'a (dyn Fn(u64, u64) -> U256 + Sync),
}

/// The noise of an encryption: the rounded Gaussian (fixed bytes per sample) or FACCT.
pub enum Noise<'a> {
    Rounded,
    Facct(FacctNoise<'a>),
}

/// encrypt_chunk_with for FACCT, drawing getrandbits from `bits` as the
/// Python sampler draws them from its source.
pub fn encrypt_chunk_facct<W: Word>(
    ring: &Ring<W>, key: &EncKey<W>, w: &[W], noise: &FacctNoise, bits: &mut dyn RandBits, ct0: &mut [W],
    ct1: &mut [W],
) {
    encrypt_chunk_with(ring, key, w, |wide, out| {
        if wide {
            facct::sample(noise.sigma3, out, bits, noise.exact3)
        } else {
            facct::sample(noise.sigma2, out, bits, noise.exact2)
        }
    }, ct0, ct1);
}

/// Random bytes per chunk of an eta = 1 tensor encryption.
pub fn tensor_chunk_bytes(n: usize) -> usize {
    gauss_bytes(n) * 3
}

/// Errors of the tensor kernels.
#[derive(Debug)]
pub enum Error {
    Input(&'static str),
    Random(std::io::Error),
}

/// Random bytes of a tensor call: given (tests), from the OS (filled per chunk
/// inside the workers), or from a stream per chunk -- `fill(c, offset, buf)` fills
/// buf with the bytes at `offset` (a multiple of 4) of chunk c's stream, which no
/// other chunk uses (the binding's per-call DRBG).
pub enum Randomness<'a> {
    Given(&'a [u8]),
    Os,
    Stream(&'a (dyn Fn(usize, u64, &mut [u8]) + Sync)),
}

/// Block of the FACCT byte readers (a multiple of 4: stream offsets).
const FACCT_BLOCK: usize = 4096;

/// Run `work(j, bits)` for every chunk with its FACCT bit source: given bytes
/// are one stream read by the chunks in order (as the Python reference reads
/// its source); the OS and the per-chunk DRBG streams give each worker its own.
pub(crate) fn facct_chunks<G>(chunks: usize, threads: usize, rnd: &Randomness, work: G) -> Result<(), Error>
where
    G: Fn(usize, &mut dyn RandBits) + Sync,
{
    match rnd {
        Randomness::Given(b) => {
            let mut pos = 0usize;
            let mut bits = ByteBits::new(FACCT_BLOCK, |buf: &mut [u8]| {
                let k = buf.len().min(b.len() - pos);
                buf[..k].copy_from_slice(&b[pos..pos + k]);
                pos += k;
                k
            });
            for j in 0..chunks {
                work(j, &mut bits);
            }
            if bits.failed {
                return Err(Error::Input("not enough random bytes"));
            }
            Ok(())
        }
        Randomness::Os | Randomness::Stream(_) => {
            let failed = std::sync::Mutex::new(false);
            parallel_for(chunks, threads, |j| {
                let mut offset = 0u64;
                let mut bits = ByteBits::new(FACCT_BLOCK, |buf: &mut [u8]| match rnd {
                    Randomness::Stream(fill) => {
                        fill(j, offset, buf);
                        offset += buf.len() as u64;
                        buf.len()
                    }
                    _ => if osrand::fill(buf).is_ok() { buf.len() } else { 0 },
                });
                work(j, &mut bits);
                if bits.failed {
                    *failed.lock().unwrap() = true;
                }
            });
            if failed.into_inner().unwrap() {
                return Err(Error::Input("the operating system's random source failed"));
            }
            Ok(())
        }
    }
}

/// Overwrite a buffer of secret bytes before it is freed.
pub fn wipe_bytes(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Overwrite a buffer of secret words before it is freed.
pub fn wipe_words<W: Word>(buf: &mut [W]) {
    for v in buf.iter_mut() {
        unsafe { std::ptr::write_volatile(v, W::ZERO) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Chunk j's random bytes: a slice of the given ones, or `own` filled from the OS
/// or the chunk's stream (the caller wipes `own` after use).
pub(crate) fn chunk_bytes<'b>(rnd: &Randomness<'b>, j: usize, per: usize, own: &'b mut Vec<u8>)
    -> Result<&'b [u8], std::io::Error>
where
    'b: 'b,
{
    match rnd {
        Randomness::Given(b) => Ok(&b[per * j..per * (j + 1)]),
        Randomness::Os => {
            own.resize(per, 0);
            osrand::fill(own)?;
            Ok(own.as_slice())
        }
        Randomness::Stream(fill) => {
            own.resize(per, 0);
            fill(j, 0, own);
            Ok(own.as_slice())
        }
    }
}

/// The pad H_c = hash_to_zq_vector(material, mode) plus u, mod q (MCFE w without x).
fn pad_row<W: Word>(ring: &Ring<W>, u: &[W], material: Option<&[u8]>, mode: HashMode, out: &mut [W]) {
    let n = ring.n;
    match material {
        Some(mat) => hash_to_zq_mode(&ring.m, mat, mode, &mut out[..n]),
        None => out.iter_mut().for_each(|v| *v = W::ZERO),
    }
    for i in 0..n {
        out[i] = ring.m.add(u[i], out[i]);
    }
}

/// MCFE ndarray encryption of one layer (eta = 1): chunk c holds x[c n .. (c + 1) n]
/// (zero-extended) and is encrypted as w = x + u + H_c mod q. `emit(c, ct0, ct1)` is
/// called from the worker threads with each chunk's ciphertext.
pub fn encrypt_tensor<W: Word, F>(
    ring: &Ring<W>, key: &EncKey<W>, x: &[i64], materials: Option<&[&[u8]]>, sigma2: f64, sigma3: f64,
    noise: &Noise, rnd: Randomness, threads: usize, emit: F,
) -> Result<usize, Error>
where
    F: Fn(usize, &[W], &[W]) + Sync,
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
        let mut w = vec![W::ZERO; n];
        pad_row(ring, u, materials.map(|m| m[j]), key.hash_mode, &mut w);
        let k0 = j * n;
        for i in 0..n.min(x.len() - k0) {
            w[i] = ring.m.add(w[i], ring.m.reduce_i128(x[k0 + i] as i128));
        }
        w
    };
    if let Noise::Facct(nz) = noise {
        facct_chunks(chunks, threads, &rnd, |j, bits| {
            let (mut c0, mut c1) = (vec![W::ZERO; n], vec![W::ZERO; n]);
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
        let (mut c0, mut c1) = (vec![W::ZERO; n], vec![W::ZERO; n]);
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

/// decode_scaled of lwe_utils: v mod q to the signed mu closest to v / Delta;
/// None if |mu| > bound; Err(()) if |mu| does not fit an i64.
#[inline]
pub fn decode_scaled(v: u128, q: u128, delta: u128, bound: i128) -> Result<Option<i64>, ()> {
    if delta == 0 {
        return Err(()); // not a valid Delta (the bindings refuse it before)
    }
    let half = q >> 1;
    let sv: i128 = if v > half { v as i128 - q as i128 } else { v as i128 };
    let den = 2 * delta as i128;
    let num = 2 * sv + delta as i128;
    let mut mu = num / den;
    if num % den != 0 && num < 0 {
        mu -= 1;
    }
    if mu > bound || mu < -bound {
        return Ok(None);
    }
    if mu > i64::MAX as i128 || mu < i64::MIN as i128 {
        return Err(());
    }
    Ok(Some(mu as i64))
}

/// floor(num / den) if below 2^127 (den > 0): shift-and-subtract over the quotient's bits only.
fn small_quotient<W: Word>(num: W, den: W) -> Option<i128> {
    if num < den {
        return Some(0);
    }
    let shift = num.bits() - den.bits();
    if shift >= 127 {
        return None; // quotient >= 2^126 ... beyond any bound below
    }
    let (mut r, mut d, mut quot) = (num, den.shl(shift), 0u128);
    for _ in 0..=shift {
        quot <<= 1;
        if r >= d {
            r = r.wsub(d);
            quot |= 1;
        }
        d = d.shr1();
    }
    i128::try_from(quot).ok()
}

/// decode_scaled on words of any size (q < 2^MAX_Q_BITS, so 2 q fits a word);
/// bound < 2^126.
#[inline]
pub fn decode_scaled_w<W: Word>(v: W, q: W, delta: W, bound: i128) -> Result<Option<i64>, ()> {
    if W::BITS <= 128 {
        return decode_scaled(v.to_u128(), q.to_u128(), delta.to_u128(), bound);
    }
    if delta == W::ZERO {
        return Err(());
    }
    let den = delta.shl1();
    // v <= q / 2: mu = floor((2 v + Delta) / (2 Delta)) >= 0; else v - q = -a and
    // mu = floor((Delta - 2 a) / (2 Delta)) = -ceil((2 a - Delta) / (2 Delta)) <= 0
    let quot = if v <= q.shr1() {
        small_quotient(v.shl1().wadd(delta), den)
    } else {
        let a2 = q.wsub(v).shl1();
        if a2 <= delta { Some(0) } else { small_quotient(a2.wsub(delta).wadd(den).wsub(W::ONE), den).map(|m| -m) }
    };
    let mu = match quot {
        Some(m) => m,
        None => return Ok(None), // |mu| >= 2^126 > bound
    };
    if mu > bound || mu < -bound {
        return Ok(None);
    }
    if mu > i64::MAX as i128 || mu < i64::MIN as i128 {
        return Err(());
    }
    Ok(Some(mu as i64))
}

/// MIFERingLWE._decrypt_with_pad_term of one chunk: decode_scaled of
/// -Delta z + sum_s (<ct0_s, y_s> - ct1_s sk_s), the products summed in the NTT
/// domain. `slots[s]` holds the eta rows of ct0 and ct1 ((eta + 1) n words),
/// `ym[s * eta + r]` the weights in Montgomery form, `sk_hats[s]` NTT(sk_s) in
/// Montgomery form. Writes k values and their None flags; false if a value
/// does not fit an i64.
pub fn decrypt_chunk<W: Word>(
    ring: &Ring<W>, eta: usize, z: &[W], slots: &[&[W]], ym: &[W], sk_hats: &[&[W]], delta: W, bound: i128,
    out: &mut [i64], none: &mut [u8],
) -> bool {
    let n = ring.n;
    let m = &ring.m;
    let delta_m = m.to_mont(delta);
    let (mut tot, mut acc, mut xh, mut rhs) = (vec![W::ZERO; n], vec![W::ZERO; n], vec![W::ZERO; n], vec![W::ZERO; n]);
    for i in 0..n {
        tot[i] = m.sub(W::ZERO, m.mul(z[i], delta_m));
    }
    for (s, slot) in slots.iter().enumerate() {
        for r in 0..eta {
            let y = ym[s * eta + r];
            let row = &slot[r * n..(r + 1) * n];
            for i in 0..n {
                tot[i] = m.add(tot[i], m.mul(row[i], y));
            }
        }
        ring.forward(&slot[eta * n..(eta + 1) * n], &mut xh);
        ring.pw_mul_add(&mut acc, sk_hats[s], &xh);
    }
    ring.inverse(&acc, &mut rhs);
    for i in 0..out.len() {
        let v = m.sub(tot[i], rhs[i]);
        match decode_scaled_w(v, ring.q, delta, bound) {
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

/// ThresholdMCFERingLWE.share_decrypt of one chunk once its noise e is drawn:
/// w = lambda (Delta z + sum_s ct1_s sk_s) + e and c_y = sum_s sum_row
/// y_{s,row} ct0_{s,row} mod q, the products ct1_s sk_s summed in the NTT
/// domain. `slots[s]` holds the eta rows of ct0 and ct1 ((eta + 1) n words),
/// `ym[s * eta + r]` the weights and `sk_hats[s]` NTT(sk_s) (key shares), both
/// in Montgomery form.
#[allow(clippy::too_many_arguments)]
pub fn share_partial<W: Word>(
    ring: &Ring<W>, eta: usize, slots: &[&[W]], ym: &[W], sk_hats: &[&[W]], z: &[W], delta: W, lambda: W,
    noise: &[W], w: &mut [W], c_y: &mut [W],
) {
    let n = ring.n;
    let m = &ring.m;
    let (mut acc, mut xh, mut kp) = (vec![W::ZERO; n], vec![W::ZERO; n], vec![W::ZERO; n]);
    for (slot, skh) in slots.iter().zip(sk_hats) {
        ring.forward(&slot[eta * n..(eta + 1) * n], &mut xh);
        ring.pw_mul_add(&mut acc, skh, &xh);
    }
    ring.inverse(&acc, &mut kp);
    let (dm, lm) = (m.to_mont(delta), m.to_mont(lambda));
    for i in 0..n {
        let key = m.add(m.mul(z[i], dm), kp[i]);
        w[i] = m.add(m.mul(key, lm), noise[i]);
        let mut cy = W::ZERO;
        for (s, slot) in slots.iter().enumerate() {
            for r in 0..eta {
                cy = m.add(cy, m.mul(slot[r * n + i], ym[s * eta + r]));
            }
        }
        c_y[i] = cy;
    }
    // the key part: the products of the ciphertexts with the key shares
    wipe_words(&mut acc);
    wipe_words(&mut kp);
}

/// ThresholdMCFERingLWE._combine_group: c_y - sum_k w_k mod q, decoded (see
/// decode_scaled) for the first out.len() values; false if one does not fit an i64.
pub fn share_combine<W: Word>(
    m: &crate::Modulus<W>, c_y: &[W], ws: &[&[W]], delta: W, bound: i128, out: &mut [i64], none: &mut [u8],
) -> bool {
    for i in 0..out.len() {
        let v = ws.iter().fold(c_y[i], |acc, w| m.sub(acc, w[i]));
        match decode_scaled_w(v, m.q, delta, bound) {
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

/// MCFERingLWEKeyGenerator's z'' for several labels: base + sum_s sum_row
/// ym[s eta + row] H_{s,l}[row] mod q, H_{s,l} = hash_to_zq_vector of
/// materials[l][s] (eta n values, row-major) with `mode`; `emit(l, z)` from the workers.
#[allow(clippy::too_many_arguments)]
pub fn pad_terms<W: Word, F>(
    ring: &Ring<W>, base: &[W], materials: &[Vec<&[u8]>], ym: &[W], eta: usize, mode: HashMode, threads: usize,
    emit: F,
) where
    F: Fn(usize, &[W]) + Sync,
{
    let n = ring.n;
    parallel_for(materials.len(), threads, |l| {
        let mut z = base.to_vec();
        let mut h = vec![W::ZERO; n * eta];
        for (s, mat) in materials[l].iter().enumerate() {
            hash_to_zq_mode(&ring.m, mat, mode, &mut h);
            for r in 0..eta {
                let y = ym[s * eta + r];
                let hr = &h[r * n..(r + 1) * n];
                for i in 0..n {
                    z[i] = ring.m.add(z[i], ring.m.mul(hr[i], y));
                }
            }
        }
        wipe_words(&mut h); // the pads of a client's u' (and dMCFE's GDSum terms)
        emit(l, &z);
        wipe_words(&mut z);
    });
}

/// The plaintext-independent part of an MCFE ndarray encryption: per chunk
/// P = pk r + e_0 + Delta (u + H_c) and ct1 = a r + e; online ct0 = P + Delta x.
/// One-time: a chunk is wiped when used and every chunk on drop.
pub struct Precomputed<W: Word> {
    pub n: usize,
    pub data: Vec<W>,
    pub used: Vec<bool>,
}

impl<W: Word> Precomputed<W> {
    fn wipe(&mut self, j: usize) {
        let n = self.n;
        wipe_words(&mut self.data[2 * n * j..2 * n * (j + 1)]);
    }
    pub fn chunks(&self) -> usize {
        self.used.len()
    }
}

impl<W: Word> Drop for Precomputed<W> {
    fn drop(&mut self) {
        for j in 0..self.used.len() {
            self.wipe(j);
        }
    }
}

pub fn precompute<W: Word>(
    ring: &Ring<W>, key: &EncKey<W>, materials: &[&[u8]], sigma2: f64, sigma3: f64, noise: &Noise,
    rnd: Randomness, threads: usize,
) -> Result<Precomputed<W>, Error> {
    let n = ring.n;
    let u = key.u.as_ref().ok_or(Error::Input("precompute needs the pad u"))?;
    if key.eta != 1 {
        return Err(Error::Input("precompute needs eta = 1"));
    }
    let chunks = materials.len();
    let data: Vec<std::sync::Mutex<Vec<W>>> = (0..chunks).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let pad = |j: usize| {
        let mut w = vec![W::ZERO; n];
        pad_row(ring, u, Some(materials[j]), key.hash_mode, &mut w);
        w
    };
    let collect = |data: Vec<std::sync::Mutex<Vec<W>>>| {
        let mut flat = Vec::with_capacity(2 * n * chunks); // no reallocation: nothing left behind
        for d in data {
            let mut pc = d.into_inner().unwrap();
            flat.extend_from_slice(&pc);
            wipe_words(&mut pc);
        }
        Precomputed { n, data: flat, used: vec![false; chunks] }
    };
    if let Noise::Facct(nz) = noise {
        facct_chunks(chunks, threads, &rnd, |j, bits| {
            let mut pc = vec![W::ZERO; 2 * n];
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
        let mut pc = vec![W::ZERO; 2 * n];
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

/// Online part for one layer: chunks first .. first + ceil(L / n) - 1, then wiped.
pub fn encrypt_precomputed<W: Word, F>(
    ring: &Ring<W>, key: &EncKey<W>, pre: &mut Precomputed<W>, first: usize, x: &[i64], threads: usize, emit: F,
) -> Result<usize, Error>
where
    F: Fn(usize, &[W], &[W]) + Sync,
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
        parallel_for(chunks, threads, |j| {
            let jj = first + j;
            let mut c0 = data[2 * n * jj..2 * n * jj + n].to_vec();
            let c1 = &data[2 * n * jj + n..2 * n * (jj + 1)];
            let k0 = j * n;
            for i in 0..n.min(x.len() - k0) {
                c0[i] = ring.m.add(c0[i], ring.m.mul(ring.m.reduce_i128(x[k0 + i] as i128), key.delta_m));
            }
            emit(j, &c0, c1);
            for v in c0.iter_mut() {
                unsafe { std::ptr::write_volatile(v, W::ZERO) };
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
    use crate::arith::U256;

    /// Python's decode_scaled on i128 (exact for the moduli used here).
    fn py_decode(v: i128, q: i128, delta: i128, bound: i128) -> Option<i128> {
        let v = if v > q / 2 { v - q } else { v };
        let mu = (2 * v + delta).div_euclid(2 * delta);
        if mu.abs() > bound { None } else { Some(mu) }
    }

    #[test]
    fn wide_decode_matches_python() {
        let q: i128 = (1 << 100) + 277; // any odd q
        let delta = q / 1009;
        let qw = U256::from_u128(q as u128);
        let dw = U256::from_u128(delta as u128);
        let mut vals: Vec<i128> = vec![0, 1, delta / 2, delta / 2 + 1, delta, q / 2, q / 2 + 1, q - 1, q - delta / 2,
                                       q - delta / 2 - 1, q - delta, q - delta - 1];
        let mut s = 12345u128;
        for _ in 0..2000 {
            s = s.wrapping_mul(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f).wrapping_add(7);
            vals.push((s % q as u128) as i128);
        }
        for bound in [0i128, 3, 504, 505, 1 << 70] {
            for &v in &vals {
                let want = py_decode(v, q, delta, bound).map(|m| m as i64);
                assert_eq!(decode_scaled_w(U256::from_u128(v as u128), qw, dw, bound), Ok(want), "v {} bound {}", v, bound);
                assert_eq!(decode_scaled(v as u128, q as u128, delta as u128, bound), Ok(want));
            }
        }
        // quotients beyond 64 bits: delta 1
        let big = U256::ONE.shl(200);
        let q2 = big.wadd(U256::from_u128(1)).shl1().wsub(U256::ONE);
        assert_eq!(decode_scaled_w(big, q2, U256::ONE, 1 << 70), Ok(None));
        assert_eq!(decode_scaled_w(big, q2, U256::ONE, 1i128 << 100), Ok(None));
        let v = U256::ONE.shl(80);
        assert_eq!(decode_scaled_w(v, q2, U256::ONE, 1i128 << 100), Err(())); // 2^80 decodes, beyond i64
        assert_eq!(decode_scaled_w(q2.wsub(v), q2, U256::ONE, 1i128 << 100), Err(()));
        assert_eq!(decode_scaled_w(U256::from_u128(5), q2, U256::ONE, 10), Ok(Some(5)));
        assert_eq!(decode_scaled_w(q2.wsub(U256::from_u128(5)), q2, U256::ONE, 10), Ok(Some(-5)));
    }
}
