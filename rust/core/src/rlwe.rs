//! BMMS21 MIFE / MCFE over Ring-LWE: the loops of pyfe4ai.schemes.mife.ring_lwe
//! and mcfe.ring_lwe on words, for one chunk or a whole tensor.

use crate::arith::Word;
use crate::gauss::{gauss_bytes, gauss_fill};
use crate::keccak::{shake256, zq_block, zq_reduce};
use crate::ntt::Ring;
use crate::{osrand, parallel_for};

/// Key of one slot: NTT(a) and NTT(pk_row) in Montgomery form, the pads u_row, Delta.
#[derive(Clone, Debug)]
pub struct EncKey<W: Word> {
    pub eta: usize,
    pub a_hat: Vec<W>,
    pub pk_hat: Vec<W>,
    pub u: Option<Vec<W>>,
    pub delta: W,
    pub delta_m: W,
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
        }
    }
}

/// One chunk of BMMS21 encryption (MIFERingLWE._encrypt_rows): r <- D(sigma2);
/// ct0_row = pk_row r + e_row + Delta w_row, e_row <- D(sigma3); ct1 = a r + e,
/// e <- D(sigma2). Random bytes are consumed for r, the e_row, e in that order.
pub fn encrypt_chunk<W: Word>(
    ring: &Ring<W>, key: &EncKey<W>, w: &[W], sigma2: f64, sigma3: f64, rnd: &[u8],
    ct0: &mut [W], ct1: &mut [W],
) {
    let n = ring.n;
    let m = &ring.m;
    let mut smp = vec![0i64; n];
    let (mut r, mut rh, mut t) = (vec![W::ZERO; n], vec![W::ZERO; n], vec![W::ZERO; n]);
    let mut pos = 0;
    gauss_fill(&rnd[pos..], sigma2, &mut smp);
    pos += gauss_bytes(n);
    for i in 0..n {
        r[i] = m.reduce_i128(smp[i] as i128);
    }
    ring.forward(&r, &mut rh);
    for row in 0..key.eta {
        let pkh = &key.pk_hat[row * n..(row + 1) * n];
        for i in 0..n {
            t[i] = m.mul(pkh[i], rh[i]);
        }
        let out = &mut ct0[row * n..(row + 1) * n];
        ring.inverse(&t, out);
        gauss_fill(&rnd[pos..], sigma3, &mut smp);
        pos += gauss_bytes(n);
        let wr = &w[row * n..(row + 1) * n];
        for i in 0..n {
            out[i] = m.add(m.add(out[i], m.reduce_i128(smp[i] as i128)), m.mul(wr[i], key.delta_m));
        }
    }
    for i in 0..n {
        t[i] = m.mul(key.a_hat[i], rh[i]);
    }
    ring.inverse(&t, ct1);
    gauss_fill(&rnd[pos..], sigma2, &mut smp);
    for i in 0..n {
        ct1[i] = m.add(ct1[i], m.reduce_i128(smp[i] as i128));
    }
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
/// inside the workers), or from a stream per chunk -- `fill(c, buf)` fills chunk
/// c's bytes from a stream no other chunk uses (the binding's per-call DRBG).
pub enum Randomness<'a> {
    Given(&'a [u8]),
    Os,
    Stream(&'a (dyn Fn(usize, &mut [u8]) + Sync)),
}

/// Overwrite a buffer of secret bytes before it is freed.
pub fn wipe_bytes(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Chunk j's random bytes: a slice of the given ones, or `own` filled from the OS
/// or the chunk's stream (the caller wipes `own` after use).
fn chunk_bytes<'b>(rnd: &Randomness<'b>, j: usize, per: usize, own: &'b mut Vec<u8>)
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
            fill(j, own);
            Ok(own.as_slice())
        }
    }
}

/// The pad H_c = hash_to_zq_vector(material) plus u, mod q (MCFE w without x).
fn pad_row<W: Word>(ring: &Ring<W>, u: &[W], material: Option<&[u8]>, out: &mut [W]) {
    let n = ring.n;
    match material {
        Some(mat) => {
            let nb = zq_block(ring.m.bits);
            let mut stream = vec![0u8; nb * n];
            shake256(mat, &mut stream);
            zq_reduce(&ring.m, &stream, nb, out);
        }
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
    rnd: Randomness, threads: usize, emit: F,
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
                let mut w = vec![W::ZERO; n];
                pad_row(ring, u, materials.map(|m| m[j]), &mut w);
                let k0 = j * n;
                let kk = n.min(x.len() - k0);
                for i in 0..kk {
                    w[i] = ring.m.add(w[i], ring.m.reduce_i128(x[k0 + i] as i128));
                }
                encrypt_chunk(ring, key, &w, sigma2, sigma3, rb, &mut c0, &mut c1);
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
        let skh = sk_hats[s];
        for i in 0..n {
            acc[i] = m.add(acc[i], m.mul(skh[i], xh[i]));
        }
    }
    ring.inverse(&acc, &mut rhs);
    let (q, d) = (ring.q, delta.to_u128());
    for i in 0..out.len() {
        let v = m.sub(tot[i], rhs[i]).to_u128();
        match decode_scaled(v, q, d, bound) {
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
/// materials[l][s] (eta n values, row-major); `emit(l, z)` from the workers.
pub fn pad_terms<W: Word, F>(
    ring: &Ring<W>, base: &[W], materials: &[Vec<&[u8]>], ym: &[W], eta: usize, threads: usize, emit: F,
) where
    F: Fn(usize, &[W]) + Sync,
{
    let n = ring.n;
    let nb = zq_block(ring.m.bits);
    parallel_for(materials.len(), threads, |l| {
        let mut z = base.to_vec();
        let mut stream = vec![0u8; nb * n * eta];
        let mut h = vec![W::ZERO; n * eta];
        for (s, mat) in materials[l].iter().enumerate() {
            shake256(mat, &mut stream);
            zq_reduce(&ring.m, &stream, nb, &mut h);
            for r in 0..eta {
                let y = ym[s * eta + r];
                let hr = &h[r * n..(r + 1) * n];
                for i in 0..n {
                    z[i] = ring.m.add(z[i], ring.m.mul(hr[i], y));
                }
            }
        }
        emit(l, &z);
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
        for v in self.data[2 * n * j..2 * n * (j + 1)].iter_mut() {
            unsafe { std::ptr::write_volatile(v, W::ZERO) };
        }
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
    ring: &Ring<W>, key: &EncKey<W>, materials: &[&[u8]], sigma2: f64, sigma3: f64, rnd: Randomness,
    threads: usize,
) -> Result<Precomputed<W>, Error> {
    let n = ring.n;
    let u = key.u.as_ref().ok_or(Error::Input("precompute needs the pad u"))?;
    if key.eta != 1 {
        return Err(Error::Input("precompute needs eta = 1"));
    }
    let chunks = materials.len();
    let per = tensor_chunk_bytes(n);
    if let Randomness::Given(b) = rnd {
        if b.len() < per * chunks {
            return Err(Error::Input("not enough random bytes"));
        }
    }
    let data: Vec<std::sync::Mutex<Vec<W>>> = (0..chunks).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let failed = std::sync::Mutex::new(None);
    parallel_for(chunks, threads, |j| {
        let mut own = Vec::new();
        let mut pc = vec![W::ZERO; 2 * n];
        match chunk_bytes(&rnd, j, per, &mut own) {
            Ok(rb) => {
                let mut w = vec![W::ZERO; n];
                pad_row(ring, u, Some(materials[j]), &mut w);
                let (c0, c1) = pc.split_at_mut(n);
                encrypt_chunk(ring, key, &w, sigma2, sigma3, rb, c0, c1);
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
    let mut flat = Vec::with_capacity(2 * n * chunks);
    for d in data {
        flat.extend_from_slice(&d.into_inner().unwrap());
    }
    Ok(Precomputed { n, data: flat, used: vec![false; chunks] })
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
    if first + chunks > pre.chunks() {
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
