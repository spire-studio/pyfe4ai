//! The rounded Gaussian of sampling.discrete_gaussian_matrix: Box-Muller as
//! random.Random.gauss on SystemRandom uniforms (7 bytes each), rounded half to
//! even. cos, sin and log are the C library's, called through pointers the
//! optimiser cannot see through (no sincos fusion, no substitutes), so that the
//! results are CPython's math module's.

type F = unsafe extern "C" fn(f64) -> f64;

extern "C" {
    fn cos(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn log(x: f64) -> f64;
}

static COS: F = cos;
static SIN: F = sin;
static LOG: F = log;

/// Random bytes for `count` samples (14 per pair).
pub fn gauss_bytes(count: usize) -> usize {
    14 * ((count + 1) / 2)
}

#[inline(always)]
fn uniform53(p: &[u8]) -> f64 {
    let mut x = 0u64;
    for &b in &p[..7] {
        x = (x << 8) | b as u64;
    }
    (x >> 3) as f64 * (1.0 / 9007199254740992.0)
}

/// `out.len()` samples of int(round(gauss(0.0, sigma))) from `buf` (gauss_bytes(len) bytes).
pub fn gauss_fill(buf: &[u8], sigma: f64, out: &mut [i64]) {
    let fcos: F = std::hint::black_box(COS);
    let fsin: F = std::hint::black_box(SIN);
    let flog: F = std::hint::black_box(LOG);
    const TWOPI: f64 = 2.0 * 3.141592653589793;
    let count = out.len();
    let mut i = 0;
    let mut p = 0;
    while i < count {
        let x2pi = uniform53(&buf[p..]) * TWOPI;
        let g2rad = (-2.0 * unsafe { flog(1.0 - uniform53(&buf[p + 7..])) }).sqrt();
        let z = unsafe { fcos(x2pi) } * g2rad;
        out[i] = (0.0 + z * sigma).round_ties_even() as i64;
        if i + 1 < count {
            let z2 = unsafe { fsin(x2pi) } * g2rad;
            out[i + 1] = (0.0 + z2 * sigma).round_ties_even() as i64;
        }
        i += 2;
        p += 14;
    }
}
