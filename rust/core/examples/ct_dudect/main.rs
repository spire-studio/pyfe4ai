//! Timing-leak tests of the kernels in the style of dudect (Reparaz, Balasch, Verbauwhede, DATE 2017):
//! two classes of secret inputs, interleaved at random, one timed batch of calls per measurement, and Welch's
//! t-test on the raw times, on times cropped at several percentiles and on the centred squares (second order).
//! |t| > 4.5 is flagged as a leak; a run that stays below says only that this many measurements showed none. On a
//! machine that scales its frequency with power, fixed vs random inputs differ by a tiny amount for any code, so
//! with enough batches every target is flagged: judge a target by its effect (the difference of the means, its
//! standard error `se`, Cohen's `d`) against that of a constant-time control (`shake256`) at the same batch count.
//!
//!     cargo run --release --example ct_dudect -- list
//!     cargo run --release --example ct_dudect -- <target> <batches> [calls per batch]
//!
//! Run the measurements of record on a quiet machine (roadmap v1.0/04: studio, with the bench lock), one thread.
//! `pow_secret_exp` is a control that must leak (square-and-multiply on a secret exponent), `shake256` one that
//! should not.
use pyfe4ai_core::rlwe::{self, EncKey, Noise, Randomness};
use pyfe4ai_core::{facct, facct_wide, garble, keccak, osrand, shamir, AnyRing, Modulus, Word, U256};

#[allow(dead_code)]
mod data {
    include!("data.rs");
}
use std::hint::black_box;
use std::time::Instant;

/// xorshift128+ seeded from the OS: the class sequence and the random inputs (not cryptographic, need not be).
struct Rng(u64, u64);
impl Rng {
    fn new() -> Self {
        let mut b = [0u8; 16];
        osrand::fill(&mut b).expect("OS randomness");
        // CT_SEED=n: a reproducible run (the same fixed inputs, e.g. to compare one script with and without DIT)
        if let Some(n) = std::env::var("CT_SEED").ok().and_then(|v| v.parse::<u64>().ok()) {
            b[..8].copy_from_slice(&n.to_le_bytes());
            b[8..].copy_from_slice(&n.wrapping_mul(0x9e3779b97f4a7c15).to_le_bytes());
        }
        let a = u64::from_le_bytes(b[..8].try_into().unwrap()) | 1;
        let c = u64::from_le_bytes(b[8..].try_into().unwrap()) | 2;
        Rng(a, c)
    }
    fn next(&mut self) -> u64 {
        let (mut s1, s0) = (self.0, self.1);
        self.0 = s0;
        s1 ^= s1 << 23;
        self.1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        self.1.wrapping_add(s0)
    }
}

/// Welford's running mean and variance of one class.
#[derive(Clone, Copy, Default)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
}
impl Moments {
    fn push(&mut self, x: f64) {
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
    }
    fn var(&self) -> f64 {
        if self.n > 1.0 { self.m2 / (self.n - 1.0) } else { 0.0 }
    }
}

/// Welch's t of two classes.
#[derive(Clone, Copy, Default)]
struct TTest([Moments; 2]);
impl TTest {
    fn push(&mut self, class: usize, x: f64) {
        self.0[class].push(x);
    }
    fn t(&self) -> f64 {
        let (a, b) = (self.0[0], self.0[1]);
        let se = (a.var() / a.n + b.var() / b.n).sqrt();
        if a.n < 2.0 || b.n < 2.0 || se == 0.0 { 0.0 } else { (a.mean - b.mean) / se }
    }
    fn n(&self) -> f64 {
        self.0[0].n + self.0[1].n
    }
}

/// A target: `prepare(class, rng)` makes the input of one batch outside the timed region, with the same work in
/// both classes (random values are drawn into a scratch buffer, then the fixed or the random ones are copied into
/// the input), and `run` is timed.
trait Target {
    fn prepare(&mut self, class: usize, rng: &mut Rng);
    fn run(&mut self);
}

/// An odd 60-bit modulus for the Montgomery and reduction targets (Montgomery needs q odd only).
const Q60: u64 = 0x0fff_ffff_ffe0_0001;

fn ntt_prime(bits: u32, n: usize) -> (U256, U256) {
    let m2 = U256::from_u128(2 * n as u128);
    let base = U256::ONE.shl(bits - 1);
    let mut c = base.wsub(base.rem(m2)).wadd(U256::ONE);
    let small = |v: u128| U256::from_u128(v);
    loop {
        let m = Modulus::<U256>::new(c);
        if [2u128, 3, 5, 7, 11, 13].iter().all(|&a| m.pow(small(a), c.wsub(U256::ONE)) == U256::ONE) {
            let mut cof = c.wsub(U256::ONE);
            for _ in 0..(2 * n).trailing_zeros() {
                cof = cof.shr1();
            }
            for g in 2u128..200 {
                let r = m.pow(small(g), cof);
                if m.pow(r, small(n as u128)) == c.wsub(U256::ONE) {
                    return (c, r);
                }
            }
        }
        c = c.wadd(m2);
    }
}

/// S1: Montgomery products by a secret factor (fixed vs random), `calls` chained products.
struct MontMul {
    m: Modulus<u64>,
    fixed: u64,
    a: u64,
    calls: usize,
}
impl Target for MontMul {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let r = rng.next() % self.m.q;
        self.a = if class == 0 { self.fixed } else { r };
    }
    fn run(&mut self) {
        let mut acc = black_box(self.m.rmod);
        let a = black_box(self.a);
        for _ in 0..self.calls {
            acc = self.m.mul(acc, a);
        }
        black_box(acc);
    }
}

/// S1 (1): reduce_i128 of noise-sized values, all non-negative (fixed) vs random signs.
struct ReduceI128 {
    /// Whether the random class draws signed values (else non-negative ones, as the fixed class of `reduce_i128`).
    signed: bool,
    /// `reduce_i128_mirror`: class 1 is the fixed values negated (fixed vs fixed, the signs only differ).
    mirror: Option<Vec<i64>>,
    /// `reduce_i128_mixed`: both classes random, class 0 non-negative, class 1 signed (no fixed class).
    mixed: Option<Vec<i64>>,
    scratch: Vec<i64>,
    m: Modulus<u64>,
    fixed: Vec<i64>,
    v: Vec<i64>,
}
impl Target for ReduceI128 {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        // the same work in both classes (random values drawn, then the fixed ones copied over for class 0)
        for x in self.scratch.iter_mut() {
            *x = if self.signed { (rng.next() % (1 << 20)) as i64 - (1 << 19) } else { (rng.next() % (1 << 19)) as i64 };
        }
        if let Some(pos) = self.mixed.as_mut() {
            // both classes random: a non-negative draw as well, every time, for class 0
            for x in pos.iter_mut() {
                *x = (rng.next() % (1 << 19)) as i64;
            }
            self.v.copy_from_slice(if class == 0 { self.mixed.as_ref().unwrap() } else { &self.scratch });
            return;
        }
        let other = self.mirror.as_ref().unwrap_or(&self.scratch);
        self.v.copy_from_slice(if class == 0 { &self.fixed } else { other });
    }
    fn run(&mut self) {
        let mut acc = 0u64;
        for &x in black_box(&self.v).iter() {
            acc = acc.wrapping_add(self.m.reduce_i128(x as i128));
        }
        black_box(acc);
    }
}

/// S1 (1), seeded: the input of both classes is drawn by the same code into the same buffer, from a seed that is
/// the same every time for a fixed class and fresh for a random one; no class reads another buffer (the copies of
/// the other targets leave class-dependent cache and prefetcher state, which the reduce_i128 runs on studio showed
/// as effects of either sign). `seeded_fixed_vs_signed`: class 0 fixed non-negative, class 1 random signed (the
/// reduce_i128 contrast); `seeded_pos_vs_signed`: random non-negative vs random signed; `seeded_null`: random signed
/// in both (A/A).
struct ReduceSeeded {
    m: Modulus<u64>,
    v: Vec<i64>,
    fixed_seed: u64,
    class0_fixed: bool,
    class0_signed: bool,
    class1_signed: bool,
}
impl Target for ReduceSeeded {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let fresh = rng.next() | 1;
        let mut state = if class == 0 && self.class0_fixed { self.fixed_seed } else { fresh };
        let signed = if class == 0 { self.class0_signed } else { self.class1_signed };
        let (range, offset) = if signed { (1u64 << 20, 1i64 << 19) } else { (1u64 << 19, 0) };
        for x in self.v.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *x = (state % range) as i64 - offset;
        }
    }
    fn run(&mut self) {
        let mut acc = 0u64;
        for &x in black_box(&self.v).iter() {
            acc = acc.wrapping_add(self.m.reduce_i128(x as i128));
        }
        black_box(acc);
    }
}

/// S1 (2): reduce_u128 of values below q (fixed) vs random 100-bit values (mostly the `%` path).
struct ReduceU128 {
    scratch: Vec<u128>,
    m: Modulus<u64>,
    fixed: Vec<u128>,
    v: Vec<u128>,
}
impl Target for ReduceU128 {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        for x in self.scratch.iter_mut() {
            *x = ((rng.next() as u128) << 36) ^ rng.next() as u128;
        }
        self.v.copy_from_slice(if class == 0 { &self.fixed } else { &self.scratch });
    }
    fn run(&mut self) {
        let mut acc = 0u64;
        for &x in black_box(&self.v).iter() {
            acc = acc.wrapping_add(self.m.reduce_u128(x));
        }
        black_box(acc);
    }
}

/// S2: forward NTT of a secret polynomial (fixed vs random), n = 1024, a 60-bit NTT prime (u64 words).
struct NttFwd {
    /// A/A test: both classes random (any |t| > 4.5 is a false positive of the harness or the machine).
    null: bool,
    scratch: Vec<u64>,
    ring: AnyRing,
    q: u64,
    fixed: Vec<u64>,
    input: Vec<u64>,
    out: Vec<u64>,
}
impl Target for NttFwd {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        for x in self.scratch.iter_mut() {
            *x = rng.next() % self.q;
        }
        self.input.copy_from_slice(if class == 0 && !self.null { &self.fixed } else { &self.scratch });
    }
    fn run(&mut self) {
        if let AnyRing::W64(r) = &self.ring {
            r.forward(black_box(&self.input), &mut self.out);
        }
        black_box(&self.out);
    }
}

/// S3: MCFE Ring-LWE layer encryption of one chunk (rlwe::encrypt_tensor: n = 1024, 60-bit NTT prime, eta = 1,
/// one pad material): the plaintext is the secret (fixed vs random values in [-100, 100]); r and the noise are fresh
/// OS randomness in both classes (rounded sampler).
struct RlweEncrypt {
    scratch: Vec<i64>,
    ring: AnyRing,
    key: EncKey<u64>,
    fixed: Vec<i64>,
    x: Vec<i64>,
}
impl Target for RlweEncrypt {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        for v in self.scratch.iter_mut() {
            *v = (rng.next() % 201) as i64 - 100;
        }
        self.x.copy_from_slice(if class == 0 { &self.fixed } else { &self.scratch });
    }
    fn run(&mut self) {
        if let AnyRing::W64(r) = &self.ring {
            let material: &[u8] = b"compat label material";
            let out = rlwe::encrypt_tensor(r, &self.key, black_box(&self.x), Some(&[material]), 4.97e9, 7.03e9,
                                           &Noise::Rounded, Randomness::Os, 1, |_, a, b| { black_box((a, b)); });
            assert!(out.is_ok(), "encrypt_tensor: {:?}", out.err());
        }
    }
}

/// S4: Shamir sharing (t = 3, five points) of 256 secrets mod an odd 60-bit q: fixed vs random secrets; the random
/// coefficients come from a fast generator in both classes.
struct ShamirShare {
    scratch: Vec<u64>,
    m: Modulus<u64>,
    fixed: Vec<u64>,
    secrets: Vec<u64>,
    seed: u64,
}
impl Target for ShamirShare {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        for v in self.scratch.iter_mut() {
            *v = rng.next() % self.m.q;
        }
        self.secrets.copy_from_slice(if class == 0 { &self.fixed } else { &self.scratch });
        self.seed = rng.next() | 1;
    }
    fn run(&mut self) {
        let mut state = self.seed;
        let fill = |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *b = state as u8;
            }
            true
        };
        black_box(shamir::share(&self.m, black_box(&self.secrets), 3, &[1, 2, 3, 4, 5], fill));
    }
}

/// Control that must leak: b^e mod q by square-and-multiply, e of Hamming weight 1 (fixed) vs random.
struct PowSecret {
    m: Modulus<u64>,
    e: u64,
    calls: usize,
}
impl Target for PowSecret {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let r = (rng.next() >> 6) | (1 << 58);
        self.e = if class == 0 { 1 << 58 } else { r };
    }
    fn run(&mut self) {
        let mut acc = 0u64;
        for i in 0..self.calls {
            acc ^= self.m.pow(black_box(3 + i as u64), black_box(self.e));
        }
        black_box(acc);
    }
}

/// Control that should not leak: SHAKE-256 of 64 bytes (fixed vs random) into 512 bytes.
struct Shake {
    fixed: [u8; 64],
    scratch: [u8; 64],
    msg: [u8; 64],
    out: [u8; 512],
    calls: usize,
}
impl Target for Shake {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        for c in self.scratch.chunks_mut(8) {
            c.copy_from_slice(&rng.next().to_le_bytes());
        }
        self.msg = if class == 0 { self.fixed } else { self.scratch };
    }
    fn run(&mut self) {
        for _ in 0..self.calls {
            keccak::shake256(black_box(&self.msg), &mut self.out);
        }
        black_box(&self.out);
    }
}

/// xorshift64 from a seed: the input generator of the seeded targets (the same code for both classes; a fixed class
/// reuses one seed, a random class draws a fresh one).
fn fill_seeded(mut state: u64, out: &mut [u8]) {
    for b in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = (state >> 24) as u8;
    }
}

fn seed_of(class: usize, fixed: u64, rng: &mut Rng) -> u64 {
    let fresh = rng.next() | 1;
    if class == 0 { fixed } else { fresh }
}

fn u256_dec(s: &str) -> U256 {
    s.bytes().fold(U256::ZERO, |acc, d| acc.mul_small_add(10, (d - b'0') as u64).0)
}

/// Counts a FACCT sample's getrandbits calls by kind: u and v (PRECISION bits each), y (y_bits), the sign (1 bit).
struct Counted<'a> {
    inner: &'a mut facct::XofBits,
    wide: u64,
    y: u64,
    sign: u64,
}
impl Counted<'_> {
    fn count(&mut self, k: u32) {
        match k {
            facct::PRECISION => self.wide += 1,
            1 => self.sign += 1,
            _ => self.y += 1,
        }
    }
}
impl facct::RandBits for Counted<'_> {
    fn getrandbits(&mut self, k: u32) -> U256 {
        self.count(k);
        self.inner.getrandbits(k)
    }
    fn getrandbits_into(&mut self, k: u32, out: &mut [u64]) {
        self.count(k);
        self.inner.getrandbits_into(k, out)
    }
}

/// The shape of the FACCT sample of a seed: (candidates, y draws, sign draws, outputs with y = 0, that is,
/// multiples of k). A rejection sampler's time follows the first three, which do not depend on the values returned.
fn facct_shape(table: &facct::Table, seed: u64, out: &mut [i64]) -> (u64, u64, u64, u64) {
    let mut raw = [0u8; 32];
    fill_seeded(seed, &mut raw);
    let mut xof = facct::XofBits::new(&raw);
    let mut bits = Counted { inner: &mut xof, wide: 0, y: 0, sign: 0 };
    facct::sample(table, out, &mut bits, &|_, _| U256::ZERO);
    let y0 = out.iter().filter(|&&z| z % table.k as i64 == 0).count() as u64;
    (bits.wide / 2, bits.y, bits.sign, y0)
}

/// Seeds bucketed by their FACCT shape (FACCT_SHAPE_SEEDS seeds, 40 000 by default).
fn facct_buckets(table: &facct::Table, rng: &mut Rng) -> std::collections::BTreeMap<(u64, u64, u64, u64), Vec<u64>> {
    let n: usize = std::env::var("FACCT_SHAPE_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(40_000);
    let mut out = vec![0i64; 1024];
    let mut buckets = std::collections::BTreeMap::new();
    for _ in 0..n {
        let seed = rng.next() | 1;
        buckets.entry(facct_shape(table, seed, &mut out)).or_insert_with(Vec::new).push(seed);
    }
    buckets
}

/// S5: the whole FACCT sample of 1024 noise values (sigma = 1000: the base CDT, y below k, the acceptance test),
/// the bit source a SHAKE stream from a 32-byte seed that is the secret (fixed vs random seed; one buffer).
struct FacctSample {
    table: facct::Table,
    fixed: u64,
    /// The seeds of class 1 when it is not a fresh seed: `facct_equal_trials`, one seed whose sample takes as many
    /// getrandbits calls as the fixed one; `facct_equal_shape`, the other seeds of the fixed seed's shape (as many
    /// candidates, y draws, sign draws and y = 0 outputs); `facct_y0_contrast`, seeds of the fixed seed's
    /// candidates, y draws and sign draws whose outputs have more y = 0 values. Class 1 picks one at random.
    pool: Option<Vec<u64>>,
    seed: [u8; 32],
    out: Vec<i64>,
    deferred: std::rc::Rc<std::cell::Cell<u64>>,
}
impl Target for FacctSample {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let s = match &self.pool {
            Some(pool) => {
                let i = (rng.next() % pool.len() as u64) as usize;
                if class == 0 { self.fixed } else { pool[i] }
            }
            None => seed_of(class, self.fixed, rng),
        };
        fill_seeded(s, &mut self.seed);
    }
    fn run(&mut self) {
        let mut bits = facct::XofBits::new(black_box(&self.seed));
        let deferred = self.deferred.clone();
        // the exact threshold is asked for in a 2^-100 band only: counted, never expected
        let exact = move |_x: u64, _y: u64| {
            deferred.set(deferred.get() + 1);
            U256::ZERO
        };
        facct::sample(&self.table, &mut self.out, &mut bits, &exact);
        black_box(&self.out);
    }
}

/// A bit source that replays a script of getrandbits results (one U256 per call, in order).
struct Scripted<'a> {
    vals: &'a [U256],
    pos: usize,
}
impl facct::RandBits for Scripted<'_> {
    fn getrandbits(&mut self, _k: u32) -> U256 {
        let v = self.vals[self.pos];
        self.pos += 1;
        v
    }
    fn getrandbits_into(&mut self, _k: u32, out: &mut [u64]) {
        let v = self.getrandbits(0);
        out.iter_mut().zip(v.0.iter()).for_each(|(o, w)| *o = *w);
    }
}

/// The getrandbits results of one sample of `count` values from `seed`, with their sizes.
fn facct_script(table: &facct::Table, seed: u64, count: usize) -> Vec<(u32, U256)> {
    struct Rec<'a>(&'a mut facct::XofBits, Vec<(u32, U256)>);
    impl facct::RandBits for Rec<'_> {
        fn getrandbits(&mut self, k: u32) -> U256 {
            let v = self.0.getrandbits(k);
            self.1.push((k, v));
            v
        }
        fn getrandbits_into(&mut self, _k: u32, _out: &mut [u64]) {
            unreachable!("narrow kernel")
        }
    }
    let mut raw = [0u8; 32];
    fill_seeded(seed, &mut raw);
    let mut xof = facct::XofBits::new(&raw);
    let mut rec = Rec(&mut xof, Vec::new());
    let mut out = vec![0i64; count];
    facct::sample(table, &mut out, &mut rec, &|_, _| U256::ZERO);
    rec.1
}

/// A uniform value below 2^bits (bits <= 192).
fn rand_bits(rng: &mut Rng, bits: u32) -> U256 {
    let mut w = [rng.next(), rng.next(), rng.next(), 0];
    for (i, x) in w.iter_mut().enumerate().take(3) {
        let lo = 64 * i as u32;
        if bits <= lo {
            *x = 0;
        } else if bits < lo + 64 {
            *x &= (1u64 << (bits - lo)) - 1;
        }
    }
    U256(w)
}

/// Another script with the same decisions as `script` -- the same calls, the same y draws redrawn, the same
/// candidates rejected, the same -0 redraws -- and new values: u, y and the sign drawn afresh within those
/// decisions, v on the same side of the new candidate's threshold (with a margin far above the band where the
/// sampler defers to the exact threshold). The time of a sample of either script then differs only by the values.
fn facct_same_decisions(table: &facct::Table, script: &[(u32, U256)], rng: &mut Rng, out: &mut Vec<U256>) {
    let (k, yb) = (table.k, table.y_bits);
    assert!(yb > 1, "the parse tells y draws from signs by their sizes");
    let c = table.c_mant as f64 * 2f64.powi(table.c_exp);
    let x_of = |u: &U256| table.cdt.iter().filter(|t| u >= t).count() as u64;
    out.clear();
    let mut i = 0;
    while i < script.len() {
        assert_eq!(script[i].0, facct::PRECISION);
        let u0 = script[i].1;
        let mut j = i + 1;
        let mut ydraws = Vec::new();
        while script[j].0 == yb {
            ydraws.push(script[j].1.0[0]);
            j += 1;
        }
        assert_eq!(script[j].0, facct::PRECISION); // v
        let accepted = j + 1 < script.len() && script[j + 1].0 == 1;
        let (y0, x0) = (*ydraws.last().unwrap(), x_of(&u0));
        let minus_zero = accepted && x0 == 0 && y0 == 0 && script[j + 1].1.0[0] == 1;
        // the new candidate: (u, y) with the old decisions possible, then v and the sign
        let (u, y, v) = loop {
            let (u, y) = if minus_zero {
                let mut u = rand_bits(rng, facct::PRECISION);
                while x_of(&u) != 0 {
                    u = rand_bits(rng, facct::PRECISION);
                }
                (u, 0)
            } else {
                (rand_bits(rng, facct::PRECISION), rng.next() % k)
            };
            if y == 0 {
                if accepted {
                    break (u, 0, rand_bits(rng, facct::PRECISION));
                }
                continue; // y = 0 is always accepted
            }
            let x = x_of(&u);
            let p = (-c * (y as f64) * (y as f64 + 2.0 * k as f64 * x as f64)).exp(); // threshold / 2^192
            let t = p * 18446744073709551616.0; // its top 64 bits
            let (lo, hi) = if accepted { (0.0, t.floor() - 1048576.0) } else { (t.ceil() + 1048576.0, 18446744073709551615.0) };
            if hi <= lo {
                continue;
            }
            let top = lo as u64 + rng.next() % ((hi - lo) as u64);
            let mut v = rand_bits(rng, 128);
            v.0[2] = top;
            break (u, y, v);
        };
        out.push(u);
        for (n, _) in ydraws.iter().enumerate() {
            out.push(if n + 1 < ydraws.len() {
                U256([k + rng.next() % ((1u64 << yb) - k), 0, 0, 0]) // a redraw: y >= k
            } else {
                U256([y, 0, 0, 0])
            });
        }
        out.push(v);
        i = j + 1;
        if accepted {
            let z = k * x_of(&u) + y;
            let sign = if minus_zero { 1 } else if z == 0 { 0 } else { rng.next() & 1 };
            out.push(U256([sign, 0, 0, 0]));
            i += 1;
        }
    }
    assert_eq!(out.len(), script.len());
}

/// Another script with the same decisions whose values follow the sampler's own conditional distributions: every
/// candidate is drawn afresh from the proposal (u, y uniform below k, v, the sign) until its decision -- accepted
/// or rejected by the acceptance test (an f64 threshold, wrong only within 2^-50 of it), and -0 or not -- is the
/// old one. Real scripts and these have the same distribution, so a fixed real script can be compared with them.
fn facct_conditional(table: &facct::Table, script: &[(u32, U256)], rng: &mut Rng, out: &mut Vec<U256>) {
    let (k, yb) = (table.k, table.y_bits);
    let c = table.c_mant as f64 * 2f64.powi(table.c_exp);
    let x_of = |u: &U256| table.cdt.iter().filter(|t| u >= t).count() as u64;
    out.clear();
    let mut i = 0;
    while i < script.len() {
        let x0 = x_of(&script[i].1);
        let mut j = i + 1;
        let mut ydraws = 0;
        while script[j].0 == yb {
            ydraws += 1;
            j += 1;
        }
        let y0 = script[j - 1].1.0[0];
        let accepted = j + 1 < script.len() && script[j + 1].0 == 1;
        let minus_zero = accepted && x0 == 0 && y0 == 0 && script[j + 1].1.0[0] == 1;
        let (u, y, v, sign) = loop {
            let (u, y, v, sign) = (rand_bits(rng, facct::PRECISION), rng.next() % k, rand_bits(rng, facct::PRECISION),
                                   rng.next() & 1);
            let x = x_of(&u);
            let p = (-c * (y as f64) * (y as f64 + 2.0 * k as f64 * x as f64)).exp();
            let acc = y == 0 || (v.0[2] as f64) < p * 18446744073709551616.0;
            let mz = acc && x == 0 && y == 0 && sign == 1;
            if acc == accepted && mz == minus_zero {
                break (u, y, v, sign);
            }
        };
        out.push(u);
        for _ in 1..ydraws {
            out.push(U256([k + rng.next() % ((1u64 << yb) - k), 0, 0, 0]));
        }
        out.push(U256([y, 0, 0, 0]));
        out.push(v);
        i = j + 1;
        if accepted {
            out.push(U256([sign, 0, 0, 0]));
            i += 1;
        }
    }
    assert_eq!(out.len(), script.len());
}

/// Diagnostics of facct_values_cond: the seed's own script with, for `mix` = "low", the low 128 bits of every u and
/// v drawn again (x and the decisions stay, but for a tie in the top word, about 2^-64); "vtop", the top word of v
/// drawn again on the same side of the candidate's threshold (an f64 estimate, 2^20 away from it).
fn facct_mix(table: &facct::Table, script: &[(u32, U256)], base: &[U256], rng: &mut Rng, out: &mut Vec<U256>,
             mix: &str) {
    // `base`: class 0's values (the seed's own or drawn ones), with the calls and decisions of `script`; "none"
    // copies it unchanged (an A/A of that script)
    let (k, yb) = (table.k, table.y_bits);
    let c = table.c_mant as f64 * 2f64.powi(table.c_exp);
    let x_of = |u: &U256| table.cdt.iter().filter(|t| u >= t).count() as u64;
    out.clear();
    out.extend_from_slice(base);
    let mut i = 0;
    while i < script.len() {
        let x = x_of(&base[i]);
        let mut j = i + 1;
        while script[j].0 == yb {
            j += 1;
        }
        let y = base[j - 1].0[0];
        let accepted = j + 1 < script.len() && script[j + 1].0 == 1;
        match mix {
            "low" => {
                for at in [i, j] {
                    out[at].0[0] = rng.next();
                    out[at].0[1] = rng.next();
                }
            }
            "vtop" if y != 0 => {
                let p = (-c * (y as f64) * (y as f64 + 2.0 * k as f64 * x as f64)).exp();
                let t = p * 18446744073709551616.0;
                let (lo, hi) = if accepted { (0.0, t.floor() - 1048576.0) } else { (t.ceil() + 1048576.0, 18446744073709551615.0) };
                if hi > lo {
                    out[j].0[2] = lo as u64 + rng.next() % ((hi - lo) as u64);
                }
            }
            _ => {}
        }
        i = j + 1 + accepted as usize;
    }
}

/// S5, FACCT values only: the decisions of one seed's sample (which candidates are rejected, which draws of y
/// redrawn, which -0), and values drawn within them (`facct_same_decisions`): one such script fixed for class 0,
/// a fresh one per batch for class 1, replayed from one buffer, no hashing in the timed part.
struct FacctValues {
    table: facct::Table,
    script: Vec<(u32, U256)>,
    fixed: Vec<U256>,
    fresh: Vec<U256>,
    buf: Vec<U256>,
    out: Vec<i64>,
    /// `facct_values_null`: both classes replay the fixed script (A/A)
    null: bool,
    /// FACCT_DIT=1: the timed call runs with the data-independent-timing bit (PSTATE.DIT) set
    dit: bool,
    /// `facct_values_cond`: class 1 draws from the conditional distributions (`facct_conditional`), class 0 replays
    /// the seed's own script
    conditional: bool,
}

/// Sets or clears PSTATE.DIT (ARMv8.4 data-independent timing; Apple silicon), a no-op elsewhere.
fn set_dit(on: bool) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        if on {
            core::arch::asm!(".arch_extension dit", "msr DIT, #1", options(nomem, nostack));
        } else {
            core::arch::asm!(".arch_extension dit", "msr DIT, #0", options(nomem, nostack));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = on;
}
impl Target for FacctValues {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        // the same work for both classes: draw a script with the same decisions, then copy one script into the
        // buffer the run reads (the fixed one for class 0)
        if let Ok(mix) = std::env::var("FACCT_MIX") {
            // diagnostics: the seed's own script with some parts drawn again, every decision kept
            facct_mix(&self.table, &self.script, &self.fixed, rng, &mut self.fresh, &mix);
        } else if self.conditional {
            facct_conditional(&self.table, &self.script, rng, &mut self.fresh);
        } else {
            facct_same_decisions(&self.table, &self.script, rng, &mut self.fresh);
        }
        self.buf.copy_from_slice(if class == 0 || self.null { &self.fixed } else { &self.fresh });
    }
    fn run(&mut self) {
        let dit = self.dit;
        if dit {
            set_dit(true);
        }
        let mut bits = Scripted { vals: black_box(&self.buf), pos: 0 };
        facct::sample(&self.table, &mut self.out, &mut bits, &|_, _| U256::ZERO);
        black_box(&self.out);
        if dit {
            set_dit(false);
        }
    }
}

/// The wide FACCT table of data.rs (sigma of the tMCFE standard profile, k of 199 bits).
fn facct_wide_table() -> facct_wide::WideTable {
    let mut cdt = [U256::ZERO; facct::CDT_SIZE];
    for (c, t) in cdt.iter_mut().zip(data::FACCT_WIDE_CDT.iter()) {
        *c = u256_dec(t);
    }
    facct_wide::WideTable { k: data::FACCT_WIDE_K, y_bits: data::FACCT_WIDE_Y_BITS, cdt, c_mant: data::FACCT_WIDE_C_MANT,
                            c_exp: data::FACCT_WIDE_C_EXP }
}

/// Values per wide sample (each is 8 words).
const WIDE_COUNT: usize = 256;

/// The shape of the wide FACCT sample of a seed: (candidates, y draws, sign draws); y = 0 has probability 2^-198.
fn facct_wide_shape(table: &facct_wide::WideTable, seed: u64, mags: &mut [u64], neg: &mut [u8]) -> (u64, u64, u64) {
    let mut raw = [0u8; 32];
    fill_seeded(seed, &mut raw);
    let mut xof = facct::XofBits::new(&raw);
    let mut bits = Counted { inner: &mut xof, wide: 0, y: 0, sign: 0 };
    facct_wide::sample(table, mags, neg, &mut bits, &|_, _| U256::ZERO);
    (bits.wide / 2, bits.y, bits.sign)
}

/// S5, wide kernel: seeds of one shape (FACCT_SHAPE_SEEDS seeds, 40 000 by default, bucketed by shape; the largest
/// bucket's first seed is the fixed one, class 1 picks among the rest): the values only.
struct FacctWideSample {
    table: facct_wide::WideTable,
    fixed: u64,
    pool: Vec<u64>,
    seed: [u8; 32],
    mags: Vec<u64>,
    neg: Vec<u8>,
}
impl Target for FacctWideSample {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let i = (rng.next() % self.pool.len() as u64) as usize;
        fill_seeded(if class == 0 { self.fixed } else { self.pool[i] }, &mut self.seed);
    }
    fn run(&mut self) {
        let mut bits = facct::XofBits::new(black_box(&self.seed));
        facct_wide::sample(&self.table, &mut self.mags, &mut self.neg, &mut bits, &|_, _| U256::ZERO);
        black_box((&self.mags, &self.neg));
    }
}

/// S13 (0.6): half-gates garbling of a CIRC v1 circuit (linear_threshold(4, 4, 4), mode hardwired). `garble_x`:
/// the garbler's data bits x are the secret (fixed vs random), the garbling seed fresh in both classes;
/// `garble_seed`: the 32-byte garbling seed is the secret (delta, every 0-label), x the same in both.
struct Garble {
    secret_is_x: bool,
    fixed: u64,
    fixed_x: u64,
    salt: [u8; 56],
    x: Vec<u8>,
    seed: [u8; 32],
}
impl Target for Garble {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let secret = seed_of(class, self.fixed, rng);
        let other = rng.next() | 1;
        let (xs, ss) = if self.secret_is_x { (secret, other) } else { (self.fixed_x, secret) };
        fill_seeded(xs, &mut self.x);
        fill_seeded(ss, &mut self.seed);
    }
    fn run(&mut self) {
        let g = garble::garble(data::CIRCUIT, garble::Mode::Hardwired, &self.salt, black_box(&self.x), black_box(&self.seed));
        black_box(g.map(|g| g.tables.len()).ok());
    }
}

/// S3: the whole MCFE Ring-LWE layer encryption of 4096 elements (four chunks, n = 1024, 60-bit NTT prime), the
/// plaintext the secret, drawn from a seed into one buffer (fixed vs random); r and the noise from the OS.
struct RlweTensor {
    ring: AnyRing,
    key: pyfe4ai_core::rlwe::EncKey<u64>,
    fixed: u64,
    raw: Vec<u8>,
    x: Vec<i64>,
}
impl Target for RlweTensor {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let s = seed_of(class, self.fixed, rng);
        fill_seeded(s, &mut self.raw);
        for (v, b) in self.x.iter_mut().zip(self.raw.iter()) {
            *v = (*b % 201) as i64 - 100;
        }
    }
    fn run(&mut self) {
        if let AnyRing::W64(r) = &self.ring {
            let mats: Vec<Vec<u8>> = (0..4).map(|c| format!("material {}", c).into_bytes()).collect();
            let refs: Vec<&[u8]> = mats.iter().map(|m| m.as_slice()).collect();
            let out = rlwe::encrypt_tensor(r, &self.key, black_box(&self.x), Some(&refs), 4.97e9, 7.03e9,
                                           &Noise::Rounded, Randomness::Os, 1, |_, a, b| { black_box((a, b)); });
            assert!(out.is_ok(), "encrypt_tensor: {:?}", out.err());
        }
    }
}

/// S4, seeded: Shamir sharing (t = 3, five points) of 256 secrets mod an odd 60-bit q, the secrets drawn from a
/// seed into one buffer (fixed vs random); the coefficients from a fresh seed in both classes.
struct ShamirSeeded {
    m: Modulus<u64>,
    fixed: u64,
    raw: Vec<u8>,
    secrets: Vec<u64>,
    coeff_seed: u64,
}
impl Target for ShamirSeeded {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let s = seed_of(class, self.fixed, rng);
        self.coeff_seed = rng.next() | 1;
        fill_seeded(s, &mut self.raw);
        for (v, c) in self.secrets.iter_mut().zip(self.raw.chunks(8)) {
            *v = u64::from_le_bytes(c.try_into().unwrap()) % self.m.q;
        }
    }
    fn run(&mut self) {
        let mut state = self.coeff_seed;
        let fill = |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *b = state as u8;
            }
            true
        };
        black_box(shamir::share(&self.m, black_box(&self.secrets), 3, &[1, 2, 3, 4, 5], fill));
    }
}

/// S2, seeded: the forward NTT (n = 1024, 60-bit NTT prime) of a polynomial drawn from a seed into the input buffer
/// itself (fixed vs random seed), no copy from another buffer.
struct NttSeeded {
    ring: AnyRing,
    q: u64,
    fixed: u64,
    input: Vec<u64>,
    out: Vec<u64>,
}
impl Target for NttSeeded {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        let mut state = seed_of(class, self.fixed, rng);
        for x in self.input.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *x = state % self.q;
        }
    }
    fn run(&mut self) {
        if let AnyRing::W64(r) = &self.ring {
            r.forward(black_box(&self.input), &mut self.out);
        }
        black_box(&self.out);
    }
}

const TARGETS: &[(&str, &str, usize)] = &[
    ("mont_mul_u64", "S1: Montgomery product by a secret factor", 4096),
    ("reduce_i128", "S1 (1): signed reduction of noise-sized values", 4096),
    ("reduce_i128_signs", "S1 (1), fixed vs random only: a fixed draw of signed values vs random signed ones", 4096),
    ("reduce_i128_pos", "S1 (1), no signs: fixed vs random non-negative values", 4096),
    ("reduce_i128_mirror", "S1 (1), signs only: fixed non-negative values vs the same values negated", 4096),
    ("reduce_i128_mixed", "S1 (1), both random: non-negative values vs signed ones", 4096),
    ("seeded_fixed_vs_signed", "S1 (1), one buffer, seeded: fixed non-negative vs random signed", 4096),
    ("seeded_pos_vs_signed", "S1 (1), one buffer, seeded: random non-negative vs random signed", 4096),
    ("seeded_null", "S1 (1), one buffer, seeded: random signed in both classes (A/A)", 4096),
    ("facct_sample", "S5, seeded: the whole FACCT sample of 1024 values (sigma = 1000), the bit stream secret", 1),
    ("facct_equal_trials", "S5: FACCT, two fixed seeds whose samples take as many getrandbits calls", 1),
    ("facct_equal_shape", "S5: FACCT, seeds of one shape (candidates, y draws, signs, y = 0 outputs): values only", 1),
    ("facct_y0_contrast", "S5: FACCT, one time shape; no y = 0 output vs the most found (the skipped test)", 1),
    ("facct_values", "S5: FACCT, 1024 values replayed: one seed's bits vs new values with the same decisions", 1),
    ("facct_values_null", "A/A of facct_values: the fixed script in both classes, the same preparation", 1),
    ("facct_values_cond", "facct_values with the seed's own script fixed and class 1 from the same conditional distributions", 1),
    ("facct_values_real", "facct_values with the seed's own script fixed: drawn and real values differ in distribution", 1),
    ("facct_wide_equal_shape", "S5: wide FACCT (k of 199 bits), 256 values, seeds of one shape: values only", 1),
    ("garble_x", "S13, seeded: half-gates garbling (mode hardwired), the garbler's data bits secret", 1),
    ("garble_seed", "S13, seeded: half-gates garbling, the garbling seed (delta, labels) secret", 1),
    ("rlwe_tensor", "S3, seeded: MCFE Ring-LWE encryption of 4096 elements (4 chunks), the plaintext secret", 1),
    ("shamir_seeded", "S4, seeded: Shamir sharing of 256 secrets, t = 3", 1),
    ("ntt_seeded", "S2, seeded: forward NTT, n = 1024, the polynomial drawn into the input buffer", 1),
    ("reduce_u128", "S1 (2): reduction of 128-bit values", 1024),
    ("ntt_fwd", "S2: forward NTT, n = 1024, 60-bit q", 1),
    ("rlwe_encrypt", "S3: Ring-LWE encryption of one chunk, the plaintext secret", 1),
    ("shamir_share", "S4: Shamir sharing of 256 secrets, t = 3", 1),
    ("pow_secret_exp", "control, must leak: square-and-multiply on a secret exponent", 64),
    ("shake256", "control, should not leak: SHAKE-256 of 64 bytes", 16),
    ("null_ntt", "A/A control: the forward NTT with random inputs in both classes (must not leak)", 1),
];

fn make(name: &str, calls: usize, rng: &mut Rng) -> Box<dyn Target> {
    let m = Modulus::<u64>::new(Q60);
    match name {
        "mont_mul_u64" => Box::new(MontMul { m, fixed: rng.next() % Q60, a: 0, calls }),
        // reduce_i128: fixed non-negative vs random signed values; _signs: a fixed draw of signed values vs random
        // signed ones (the same distribution: fixed vs random only); _pos: fixed vs random non-negative values
        "reduce_i128" | "reduce_i128_signs" | "reduce_i128_pos" | "reduce_i128_mirror" | "reduce_i128_mixed" => {
            let fixed_signed = name == "reduce_i128_signs";
            let fixed: Vec<i64> = (0..calls)
                .map(|_| if fixed_signed { (rng.next() % (1 << 20)) as i64 - (1 << 19) } else { (rng.next() % (1 << 19)) as i64 })
                .collect();
            let mirror = (name == "reduce_i128_mirror").then(|| fixed.iter().map(|&x| -x).collect());
            let mixed = (name == "reduce_i128_mixed").then(|| fixed.clone());
            Box::new(ReduceI128 { signed: name != "reduce_i128_pos", mirror, mixed, m, v: fixed.clone(),
                                  scratch: fixed.clone(), fixed })
        }
        "seeded_fixed_vs_signed" | "seeded_pos_vs_signed" | "seeded_null" => Box::new(ReduceSeeded {
            m,
            v: vec![0; calls],
            fixed_seed: rng.next() | 1,
            class0_fixed: name == "seeded_fixed_vs_signed",
            class0_signed: name == "seeded_null",
            class1_signed: true,
        }),
        "facct_sample" | "facct_equal_trials" | "facct_equal_shape" | "facct_y0_contrast" => {
            let mut cdt = [U256::ZERO; facct::CDT_SIZE];
            for (c, t) in cdt.iter_mut().zip(data::FACCT_CDT.iter()) {
                *c = u256_dec(t);
            }
            let table = facct::Table { k: data::FACCT_K, y_bits: data::FACCT_Y_BITS, cdt, c_mant: data::FACCT_C_MANT,
                                       c_exp: data::FACCT_C_EXP };
            let mut fixed = rng.next() | 1;
            let pool = match name {
                "facct_sample" => None,
                "facct_equal_trials" => {
                    // two seeds whose 1024 samples take as many getrandbits calls, with different samples
                    let calls = |seed: u64, out: &mut Vec<i64>| {
                        let mut raw = [0u8; 32];
                        fill_seeded(seed, &mut raw);
                        let mut bits = facct::XofBits::new(&raw);
                        facct::sample(&table, out, &mut bits, &|_, _| U256::ZERO);
                        bits.next_call()
                    };
                    let (mut out_a, mut out_b) = (vec![0i64; 1024], vec![0i64; 1024]);
                    let target = calls(fixed, &mut out_a);
                    Some(vec![loop {
                        let cand = rng.next() | 1;
                        if calls(cand, &mut out_b) == target && out_b != out_a {
                            eprintln!("facct_equal_trials: two seeds with {} draws each", target);
                            break cand;
                        }
                    }])
                }
                "facct_equal_shape" => {
                    // the largest bucket of one shape: its first seed is the fixed one, class 1 picks among the rest
                    let buckets = facct_buckets(&table, rng);
                    let (shape, seeds) = buckets.iter().max_by_key(|(_, s)| s.len()).expect("seeds");
                    assert!(seeds.len() >= 2, "no two seeds of one shape: raise FACCT_SHAPE_SEEDS");
                    eprintln!("facct_equal_shape: {} seeds of shape (candidates, y draws, signs, y = 0) = {:?} \
                               out of {} shapes", seeds.len(), shape, buckets.len());
                    fixed = seeds[0];
                    Some(seeds[1..].to_vec())
                }
                _ => {
                    // facct_y0_contrast: one shape of the time (candidates, y draws, signs); the fixed seed has
                    // no y = 0 output, class 1 the most found among the seeds of that shape
                    let buckets = facct_buckets(&table, rng);
                    let mut by_time = std::collections::BTreeMap::<(u64, u64, u64), Vec<(u64, u64)>>::new();
                    for (&(c, y, s, y0), seeds) in &buckets {
                        by_time.entry((c, y, s)).or_default().extend(seeds.iter().map(|&seed| (y0, seed)));
                    }
                    let best = by_time.iter().filter_map(|(key, v)| {
                        let zero = v.iter().find(|(y0, _)| *y0 == 0)?;
                        let top = v.iter().map(|(y0, _)| *y0).max()?;
                        let n_top = v.iter().filter(|(y0, _)| *y0 == top).count();
                        Some(((top, n_top), key, zero.1, v.iter().filter(|(y0, _)| *y0 == top).map(|&(_, s)| s).collect::<Vec<_>>()))
                    }).max_by_key(|(rank, ..)| *rank).expect("seeds");
                    let ((top, n_top), key, zero_seed, tops) = best;
                    assert!(top > 0, "no shape with outputs of y = 0: raise FACCT_SHAPE_SEEDS");
                    eprintln!("facct_y0_contrast: shape (candidates, y draws, signs) = {:?}: fixed seed 0 outputs \
                               of y = 0, class 1 {} seeds with {} each", key, n_top, top);
                    fixed = zero_seed;
                    Some(tops)
                }
            };
            Box::new(FacctSample { table, fixed, pool, seed: [0; 32], out: vec![0; 1024], deferred: Default::default() })
        }
        "facct_values" | "facct_values_null" | "facct_values_real" | "facct_values_cond" => {
            let mut cdt = [U256::ZERO; facct::CDT_SIZE];
            for (c, t) in cdt.iter_mut().zip(data::FACCT_CDT.iter()) {
                *c = u256_dec(t);
            }
            let table = facct::Table { k: data::FACCT_K, y_bits: data::FACCT_Y_BITS, cdt, c_mant: data::FACCT_C_MANT,
                                       c_exp: data::FACCT_C_EXP };
            let script = facct_script(&table, rng.next() | 1, 1024);
            // check: a script with the same decisions makes the same calls and 1024 values, its y = 0 differ
            let (mut buf, mut out, mut y0) = (Vec::new(), vec![0i64; 1024], Vec::new());
            for n in 0..20 {
                if n % 2 == 0 {
                    facct_same_decisions(&table, &script, rng, &mut buf);
                } else {
                    facct_conditional(&table, &script, rng, &mut buf);
                }
                let mut bits = Scripted { vals: &buf, pos: 0 };
                facct::sample(&table, &mut out, &mut bits, &|_, _| U256::ZERO);
                assert_eq!(bits.pos, buf.len(), "a script with the same decisions makes as many calls");
                y0.push(out.iter().filter(|&&z| z % table.k as i64 == 0).count());
            }
            // class 0 replays one script with the same decisions, drawn as class 1's are (a real script's values
            // follow the accepted and rejected distributions, the drawn ones the proposal's: compared with each
            // other, the two kinds differ by their distributions)
            // (facct_values_real: class 0 replays the real script, to show what the difference of the
            // distributions does)
            let mut base = Vec::new();
            if name == "facct_values_real" || name == "facct_values_cond" {
                base.extend(script.iter().map(|&(_, v)| v));
            } else {
                facct_same_decisions(&table, &script, rng, &mut base);
            }
            let mut bits = Scripted { vals: &base, pos: 0 };
            facct::sample(&table, &mut out, &mut bits, &|_, _| U256::ZERO);
            eprintln!("facct_values: {} calls; outputs of y = 0: fixed {}, same-decision scripts {:?}", script.len(),
                      out.iter().filter(|&&z| z % table.k as i64 == 0).count(), y0);
            let n = base.len();
            Box::new(FacctValues { table, script, fixed: base, fresh: Vec::new(), buf: vec![U256::ZERO; n], out: vec![0; 1024],
                                   null: name == "facct_values_null",
                                   dit: std::env::var("FACCT_DIT").map(|v| v == "1").unwrap_or(false),
                                   conditional: name == "facct_values_cond" })
        }
        "facct_wide_equal_shape" => {
            let table = facct_wide_table();
            let n: usize = std::env::var("FACCT_SHAPE_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(40_000);
            let (mut mags, mut neg) = (vec![0u64; facct_wide::W * WIDE_COUNT], vec![0u8; WIDE_COUNT]);
            let mut buckets = std::collections::BTreeMap::<(u64, u64, u64), Vec<u64>>::new();
            for _ in 0..n {
                let seed = rng.next() | 1;
                buckets.entry(facct_wide_shape(&table, seed, &mut mags, &mut neg)).or_default().push(seed);
            }
            let (shape, seeds) = buckets.iter().max_by_key(|(_, s)| s.len()).expect("seeds");
            assert!(seeds.len() >= 2, "no two seeds of one shape: raise FACCT_SHAPE_SEEDS");
            eprintln!("facct_wide_equal_shape: {} seeds of shape (candidates, y draws, signs) = {:?} out of {} shapes",
                      seeds.len(), shape, buckets.len());
            Box::new(FacctWideSample { table, fixed: seeds[0], pool: seeds[1..].to_vec(), seed: [0; 32], mags, neg })
        }
        "garble_x" | "garble_seed" => {
            let mut salt = [0u8; 56];
            fill_seeded(rng.next() | 1, &mut salt);
            Box::new(Garble { secret_is_x: name == "garble_x", fixed: rng.next() | 1, fixed_x: rng.next() | 1, salt,
                              x: vec![0; data::CIRCUIT_N.div_ceil(8)], seed: [0; 32] })
        }
        "rlwe_tensor" => {
            let n = 1024;
            let (q, root) = ntt_prime(60, n);
            let ring = AnyRing::new(q, n, root).expect("ring");
            let qw = q.to_u128() as u64;
            let key = match &ring {
                AnyRing::W64(r) => {
                    let a: Vec<u64> = (0..n).map(|_| rng.next() % qw).collect();
                    let pk: Vec<u64> = (0..n).map(|_| rng.next() % qw).collect();
                    let u: Vec<u64> = (0..n).map(|_| rng.next() % qw).collect();
                    pyfe4ai_core::rlwe::EncKey::new(r, &a, &[pk], Some(&[u]), qw >> 10)
                }
                _ => unreachable!("a 60-bit q has u64 words"),
            };
            Box::new(RlweTensor { ring, key, fixed: rng.next() | 1, raw: vec![0; 4096], x: vec![0; 4096] })
        }
        "ntt_seeded" => {
            let n = 1024;
            let (q, root) = ntt_prime(60, n);
            let ring = AnyRing::new(q, n, root).expect("ring");
            let q = q.to_u128() as u64;
            Box::new(NttSeeded { ring, q, fixed: rng.next() | 1, input: vec![0; n], out: vec![0; n] })
        }
        "shamir_seeded" => Box::new(ShamirSeeded { m, fixed: rng.next() | 1, raw: vec![0; 256 * 8], secrets: vec![0; 256],
                                                     coeff_seed: 1 }),
        "reduce_u128" => {
            let fixed: Vec<u128> = (0..calls).map(|_| (rng.next() % Q60) as u128).collect();
            Box::new(ReduceU128 { m, v: fixed.clone(), scratch: fixed.clone(), fixed })
        }
        "ntt_fwd" | "null_ntt" => {
            let n = 1024;
            let (q, root) = ntt_prime(60, n);
            let ring = AnyRing::new(q, n, root).expect("ring");
            let q = q.to_u128() as u64;
            assert!(matches!(ring, AnyRing::W64(_)), "a 60-bit q has u64 words");
            let fixed: Vec<u64> = (0..n).map(|_| rng.next() % q).collect();
            Box::new(NttFwd { null: name == "null_ntt", ring, q, input: fixed.clone(), scratch: fixed.clone(), fixed,
                              out: vec![0; n] })
        }
        "rlwe_encrypt" => {
            let n = 1024;
            let (q, root) = ntt_prime(60, n);
            let ring = AnyRing::new(q, n, root).expect("ring");
            let qw = q.to_u128() as u64;
            let key = match &ring {
                AnyRing::W64(r) => {
                    let a: Vec<u64> = (0..n).map(|_| rng.next() % qw).collect();
                    let pk: Vec<u64> = (0..n).map(|_| rng.next() % qw).collect();
                    let u: Vec<u64> = (0..n).map(|_| rng.next() % qw).collect();
                    EncKey::new(r, &a, &[pk], Some(&[u]), qw >> 10)
                }
                _ => unreachable!("a 60-bit q has u64 words"),
            };
            let fixed: Vec<i64> = (0..n).map(|_| (rng.next() % 201) as i64 - 100).collect();
            Box::new(RlweEncrypt { ring, key, x: fixed.clone(), scratch: fixed.clone(), fixed })
        }
        "shamir_share" => {
            let fixed: Vec<u64> = (0..256).map(|_| rng.next() % Q60).collect();
            Box::new(ShamirShare { m, secrets: fixed.clone(), scratch: fixed.clone(), fixed, seed: 1 })
        }
        "pow_secret_exp" => Box::new(PowSecret { m, e: 0, calls }),
        "shake256" => {
            let mut fixed = [0u8; 64];
            for c in fixed.chunks_mut(8) {
                c.copy_from_slice(&rng.next().to_le_bytes());
            }
            Box::new(Shake { fixed, scratch: fixed, msg: fixed, out: [0; 512], calls })
        }
        _ => {
            eprintln!("unknown target {} (see `list`)", name);
            std::process::exit(2);
        }
    }
}

/// Percentiles at which the cropped tests cut (as dudect, of the pooled warm-up times).
const CROPS: [f64; 6] = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999];
const WARMUP: usize = 2000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "list" {
        for (name, what, calls) in TARGETS {
            println!("{:16} {} (default {} calls per batch)", name, what, calls);
        }
        return;
    }
    let name = args[1].as_str();
    let batches: usize = args.get(2).map(|s| s.parse().expect("batches")).unwrap_or(100_000);
    let default_calls = TARGETS.iter().find(|t| t.0 == name).map(|t| t.2).unwrap_or(1);
    let calls: usize = args.get(3).map(|s| s.parse().expect("calls")).unwrap_or(default_calls);
    let mut rng = Rng::new();
    let mut target = make(name, calls, &mut rng);

    let mut warm: Vec<f64> = Vec::with_capacity(WARMUP);
    let mut crops = [0f64; CROPS.len()];
    let mut raw = TTest::default();
    let mut cropped = [TTest::default(); CROPS.len()];
    let mut second = TTest::default();
    let report = (batches / 10).max(1);
    println!("target {} batches {} calls/batch {}", name, batches, calls);
    for i in 0..batches + WARMUP {
        let class = (rng.next() & 1) as usize;
        target.prepare(class, &mut rng);
        let t0 = Instant::now();
        target.run();
        let dt = t0.elapsed().as_nanos() as f64;
        if i < WARMUP {
            warm.push(dt);
            if i + 1 == WARMUP {
                let mut s = warm.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for (k, p) in CROPS.iter().enumerate() {
                    crops[k] = s[((s.len() as f64 - 1.0) * p) as usize];
                }
            }
            continue;
        }
        raw.push(class, dt);
        for (k, c) in crops.iter().enumerate() {
            if dt <= *c {
                cropped[k].push(class, dt);
            }
        }
        // second order: the centred square, about the class's running mean (as dudect), not a fixed one, so
        // that a slow drift of the machine does not read as a difference of the classes' variances
        let d = dt - raw.0[class].mean;
        second.push(class, d * d);
        let done = i + 1 - WARMUP;
        if done % report == 0 || done == batches {
            let max_crop = cropped.iter().map(|t| t.t().abs()).fold(0.0, f64::max);
            let worst = raw.t().abs().max(max_crop).max(second.t().abs());
            // effect size: the difference of the means with its standard error, and Cohen's d (difference over
            // the pooled standard deviation); judge a target by these against the constant-time control's
            let (a, c) = (raw.0[0], raw.0[1]);
            let se = (a.var() / a.n + c.var() / c.n).sqrt();
            let pooled = (((a.n - 1.0) * a.var() + (c.n - 1.0) * c.var()) / (a.n + c.n - 2.0)).sqrt();
            let diff = a.mean - c.mean;
            println!(
                "{:>10} batches  mean {:>10.1} / {:>10.1} ns  t_raw {:>8.2}  max|t_crop| {:>8.2}  t_2nd {:>8.2}  \
                 diff {:+.3} ns  se {:.3} ns  d {:+.2e}  {}",
                done, a.mean, c.mean, raw.t(), max_crop, second.t(), diff, se,
                if pooled > 0.0 { diff / pooled } else { 0.0 },
                if worst > 4.5 { "LEAK (|t| > 4.5)" } else { "no leak detected yet" }
            );
        }
    }
    let _ = raw.n();
}
