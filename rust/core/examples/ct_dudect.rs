//! Timing-leak tests of the kernels in the style of dudect (Reparaz, Balasch, Verbauwhede, DATE 2017):
//! two classes of secret inputs, interleaved at random, one timed batch of calls per measurement, and Welch's
//! t-test on the raw times, on times cropped at several percentiles and on the centred squares (second order).
//! |t| > 4.5 counts as a leak; a run that stays below says only that this many measurements showed none.
//!
//!     cargo run --release --example ct_dudect -- list
//!     cargo run --release --example ct_dudect -- <target> <batches> [calls per batch]
//!
//! Run the measurements of record on a quiet machine (roadmap v1.0/04: studio, with the bench lock), one thread.
//! `pow_secret_exp` is a control that must leak (square-and-multiply on a secret exponent), `shake256` one that
//! should not.
use pyfe4ai_core::{keccak, osrand, AnyRing, Modulus, Word, U256};
use std::hint::black_box;
use std::time::Instant;

/// xorshift128+ seeded from the OS: the class sequence and the random inputs (not cryptographic, need not be).
struct Rng(u64, u64);
impl Rng {
    fn new() -> Self {
        let mut b = [0u8; 16];
        osrand::fill(&mut b).expect("OS randomness");
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

/// A target: `prepare(class, rng)` makes the input of one batch outside the timed region, `run` is timed.
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
        self.a = if class == 0 { self.fixed } else { rng.next() % self.m.q };
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
    m: Modulus<u64>,
    fixed: Vec<i64>,
    v: Vec<i64>,
}
impl Target for ReduceI128 {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        if class == 0 {
            self.v.copy_from_slice(&self.fixed);
        } else {
            for x in self.v.iter_mut() {
                *x = (rng.next() % (1 << 20)) as i64 - (1 << 19);
            }
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
    m: Modulus<u64>,
    fixed: Vec<u128>,
    v: Vec<u128>,
}
impl Target for ReduceU128 {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        if class == 0 {
            self.v.copy_from_slice(&self.fixed);
        } else {
            for x in self.v.iter_mut() {
                *x = ((rng.next() as u128) << 36) ^ rng.next() as u128;
            }
        }
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
    ring: AnyRing,
    q: u64,
    fixed: Vec<u64>,
    input: Vec<u64>,
    out: Vec<u64>,
}
impl Target for NttFwd {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        if class == 0 {
            self.input.copy_from_slice(&self.fixed);
        } else {
            for x in self.input.iter_mut() {
                *x = rng.next() % self.q;
            }
        }
    }
    fn run(&mut self) {
        if let AnyRing::W64(r) = &self.ring {
            r.forward(black_box(&self.input), &mut self.out);
        }
        black_box(&self.out);
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
        self.e = if class == 0 { 1 << 58 } else { (rng.next() >> 6) | (1 << 58) };
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
    msg: [u8; 64],
    out: [u8; 512],
    calls: usize,
}
impl Target for Shake {
    fn prepare(&mut self, class: usize, rng: &mut Rng) {
        if class == 0 {
            self.msg = self.fixed;
        } else {
            for c in self.msg.chunks_mut(8) {
                c.copy_from_slice(&rng.next().to_le_bytes());
            }
        }
    }
    fn run(&mut self) {
        for _ in 0..self.calls {
            keccak::shake256(black_box(&self.msg), &mut self.out);
        }
        black_box(&self.out);
    }
}

const TARGETS: &[(&str, &str, usize)] = &[
    ("mont_mul_u64", "S1: Montgomery product by a secret factor", 4096),
    ("reduce_i128", "S1 (1): signed reduction of noise-sized values", 4096),
    ("reduce_u128", "S1 (2): reduction of 128-bit values", 1024),
    ("ntt_fwd", "S2: forward NTT, n = 1024, 60-bit q", 1),
    ("pow_secret_exp", "control, must leak: square-and-multiply on a secret exponent", 64),
    ("shake256", "control, should not leak: SHAKE-256 of 64 bytes", 16),
];

fn make(name: &str, calls: usize, rng: &mut Rng) -> Box<dyn Target> {
    let m = Modulus::<u64>::new(Q60);
    match name {
        "mont_mul_u64" => Box::new(MontMul { m, fixed: rng.next() % Q60, a: 0, calls }),
        "reduce_i128" => {
            let fixed: Vec<i64> = (0..calls).map(|_| (rng.next() % (1 << 19)) as i64).collect();
            Box::new(ReduceI128 { m, v: fixed.clone(), fixed })
        }
        "reduce_u128" => {
            let fixed: Vec<u128> = (0..calls).map(|_| (rng.next() % Q60) as u128).collect();
            Box::new(ReduceU128 { m, v: fixed.clone(), fixed })
        }
        "ntt_fwd" => {
            let n = 1024;
            let (q, root) = ntt_prime(60, n);
            let ring = AnyRing::new(q, n, root).expect("ring");
            let q = q.to_u128() as u64;
            assert!(matches!(ring, AnyRing::W64(_)), "a 60-bit q has u64 words");
            let fixed: Vec<u64> = (0..n).map(|_| rng.next() % q).collect();
            Box::new(NttFwd { ring, q, input: fixed.clone(), fixed, out: vec![0; n] })
        }
        "pow_secret_exp" => Box::new(PowSecret { m, e: 0, calls }),
        "shake256" => {
            let mut fixed = [0u8; 64];
            for c in fixed.chunks_mut(8) {
                c.copy_from_slice(&rng.next().to_le_bytes());
            }
            Box::new(Shake { fixed, msg: fixed, out: [0; 512], calls })
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
    let mut warm_mean = [0f64; 2];
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
            warm_mean[class] += dt;
            if i + 1 == WARMUP {
                let mut s = warm.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for (k, p) in CROPS.iter().enumerate() {
                    crops[k] = s[((s.len() as f64 - 1.0) * p) as usize];
                }
                let half = WARMUP as f64 / 2.0;
                warm_mean = [warm_mean[0] / half, warm_mean[1] / half];
            }
            continue;
        }
        raw.push(class, dt);
        for (k, c) in crops.iter().enumerate() {
            if dt <= *c {
                cropped[k].push(class, dt);
            }
        }
        let d = dt - warm_mean[class];
        second.push(class, d * d);
        let done = i + 1 - WARMUP;
        if done % report == 0 || done == batches {
            let max_crop = cropped.iter().map(|t| t.t().abs()).fold(0.0, f64::max);
            let worst = raw.t().abs().max(max_crop).max(second.t().abs());
            println!(
                "{:>10} batches  mean {:>10.1} / {:>10.1} ns  t_raw {:>8.2}  max|t_crop| {:>8.2}  t_2nd {:>8.2}  {}",
                done, raw.0[0].mean, raw.0[1].mean, raw.t(), max_crop, second.t(),
                if worst > 4.5 { "LEAK (|t| > 4.5)" } else { "no leak detected yet" }
            );
        }
    }
    let _ = raw.n();
}
