//! Multi-exponentiation on BLS12-381 for the pairing-based schemes:
//! prod_i b_i^{k_i} in G1, G2 (sum_i [k_i] P_i, written additively here) and
//! GT, one base raised to many exponents, and the membership test of GT.
//!
//! G1 / G2: Straus' interleaved wNAF method over the GLV decomposition
//! k = k1 + lambda k2 (|k1|, |k2| ~ 2^128; arkworks' `GLVConfig`), whose 128
//! doublings are shared by all terms; arkworks' Pippenger (`VariableBaseMSM`,
//! which does not use the endomorphism) from `PIPPENGER_MIN` terms on. One
//! exponentiation is the one-term case (arkworks multiplies in G2 by plain
//! double-and-add over 255 bits). One base to many exponents: a table of the
//! multiples d 2^(w j) P of the base, so that each power is ~2 * 129 / w
//! additions without doublings (w by the number of exponents).
//!
//! GT: since p = u (mod r) for the curve parameter u, the Frobenius map is
//! f -> f^u on GT; with U = |u| = -u and k = k0 + k1 U + k2 U^2 + k3 U^3
//! (k_i < U < 2^64, as r < U^4), f^k = f^k0 (f^U)^k1 (f^U^2)^k2 (f^U^3)^k3
//! where f^U = conj(frob(f)), f^U^2 = frob^2(f), f^U^3 = conj(frob^3(f)):
//! a four-term Straus product over 64-bit exponents (64 cyclotomic squarings
//! instead of 255).
//!
//! Every input is an element of the prime-order group (the GLV and Frobenius
//! identities hold there only); the elements of `pairing.rs` are, by
//! construction and by the checks on deserialisation. Not constant-time.

use ark_bls12_381::{Fq12, Fr};
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::{AdditiveGroup, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, CyclotomicMultSubgroup, Field, One, PrimeField, Zero};

/// |u|, u = -0xd201000000010000 the BLS12-381 parameter.
const U: u64 = 0xd201000000010000;

/// From this many (nonzero) terms on, arkworks' Pippenger is used for G1 / G2.
pub const PIPPENGER_MIN: usize = 512;

/// Window of the wNAF digits of the curve method (8 odd multiples per point).
const CURVE_W: usize = 5;

/// Window of the wNAF digits of the GT method (4 odd powers per base).
const GT_W: usize = 4;

/// Signed digits d_j, each zero or odd with |d_j| < 2^(w-1), of k = sum_j d_j 2^j.
fn wnaf<B: BigInteger>(k: &B, w: usize) -> Vec<i8> {
    k.find_wnaf(w).expect("window in 2..64").into_iter().map(|d| d as i8).collect()
}

/// The value of an odd digit d as an index into a table of odd multiples, and its sign.
#[inline]
fn digit(d: i8) -> (usize, bool) {
    ((d.unsigned_abs() as usize) >> 1, d > 0)
}

/// Digits and a table of odd multiples [1, 3, 5, ...] (affine) of one point.
struct CurveTerm<'a, C: GLVConfig> {
    digits: Vec<i8>,
    table: &'a [Affine<C>],
}

/// Shared doubling loop of Straus' method.
fn straus_curve<C: GLVConfig>(terms: &[CurveTerm<'_, C>]) -> Projective<C> {
    let len = terms.iter().map(|t| t.digits.len()).max().unwrap_or(0);
    let mut acc = Projective::<C>::zero();
    for j in (0..len).rev() {
        acc.double_in_place();
        for t in terms {
            match t.digits.get(j) {
                Some(&d) if d != 0 => {
                    let (i, pos) = digit(d);
                    if pos {
                        acc += &t.table[i];
                    } else {
                        acc -= &t.table[i];
                    }
                }
                _ => {}
            }
        }
    }
    acc
}

/// P, 3P, ..., (2^(w-1) - 1)P for every point, and the same for phi(P)
/// (phi the GLV endomorphism), in affine form after one batch inversion.
fn glv_tables<C: GLVConfig>(points: &[Projective<C>], w: usize) -> (Vec<Affine<C>>, Vec<Affine<C>>) {
    let half = 1 << (w - 2);
    let mut t = Vec::with_capacity(points.len() * half);
    for p in points {
        let d = p.double();
        let mut cur = *p;
        for _ in 0..half {
            t.push(cur);
            cur += d;
        }
    }
    let t = Projective::normalize_batch(&t);
    let te = t.iter().map(C::endomorphism_affine).collect();
    (t, te)
}

/// The two terms (k1 on P, k2 on phi(P)) of the GLV decomposition of k.
fn glv_terms<'a, C: GLVConfig>(
    k: &C::ScalarField,
    table: &'a [Affine<C>],
    table_endo: &'a [Affine<C>],
    out: &mut Vec<CurveTerm<'a, C>>,
) {
    let ((s1, k1), (s2, k2)) = C::scalar_decomposition(*k);
    for (positive, half, tab) in [(s1, k1, table), (s2, k2, table_endo)] {
        let mut digits = wnaf(&half.into_bigint(), CURVE_W);
        if !positive {
            digits.iter_mut().for_each(|d| *d = -*d);
        }
        if !digits.is_empty() {
            out.push(CurveTerm { digits, table: tab });
        }
    }
}

/// sum_i [k_i] P_i in G1 or G2 (the shorter of the two slices bounds the sum).
pub fn curve_msm<C: GLVConfig>(points: &[Projective<C>], scalars: &[C::ScalarField]) -> Projective<C> {
    let (points, scalars): (Vec<_>, Vec<_>) = points
        .iter()
        .zip(scalars)
        .filter(|(p, k)| !p.is_zero() && !k.is_zero())
        .map(|(p, k)| (*p, *k))
        .unzip();
    if points.len() >= PIPPENGER_MIN {
        let affine = Projective::normalize_batch(&points);
        return <Projective<C> as VariableBaseMSM>::msm_unchecked(&affine, &scalars);
    }
    let half = 1 << (CURVE_W - 2);
    let (tab, tab_endo) = glv_tables(&points, CURVE_W);
    let mut terms = Vec::with_capacity(2 * points.len());
    for (i, k) in scalars.iter().enumerate() {
        glv_terms(k, &tab[i * half..(i + 1) * half], &tab_endo[i * half..(i + 1) * half], &mut terms);
    }
    straus_curve(&terms)
}

/// Signed digits d_j of k = sum_j d_j 2^(w j): the NAF (digits -1, 0, 1)
/// for w = 1, else windows of w bits with |d_j| <= 2^(w-1).
fn fixed_base_digits<B: BigInteger>(k: &B, w: usize) -> Vec<i8> {
    if w == 1 {
        return wnaf(k, 2);
    }
    let bits = k.num_bits() as usize;
    let mut out = Vec::with_capacity(bits / w + 2);
    let mut carry = 0u32;
    let mut j = 0;
    while j < bits || carry != 0 {
        let mut v = carry; // at most 2^w
        for b in 0..w {
            if j + b < bits && k.get_bit(j + b) {
                v += 1 << b;
            }
        }
        if v > 1 << (w - 1) {
            out.push((v as i32 - (1i32 << w)) as i8);
            carry = 1;
        } else {
            out.push(v as i8);
            carry = 0;
        }
        j += w;
    }
    out
}

/// Window of the fixed-base table for n powers (measured on G1 and G2).
fn fixed_base_window(n: usize) -> usize {
    match n {
        0..=11 => 1,
        12..=31 => 4,
        32..=95 => 5,
        _ => 6,
    }
}

/// [k_i] P for every k_i (one base): the multiples d 2^(w j) P (and their
/// GLV images) for |d| <= 2^(w-1) are computed once, then each power is a
/// sum of table entries along the signed digits of its GLV halves k1, k2
/// (no doublings; w = 1 is the NAF, about 86 additions per power).
pub fn curve_pow_many<C: GLVConfig>(p: &Projective<C>, scalars: &[C::ScalarField]) -> Vec<Projective<C>> {
    curve_pow_many_window(p, scalars, fixed_base_window(scalars.len()))
}

pub fn curve_pow_many_window<C: GLVConfig>(
    p: &Projective<C>,
    scalars: &[C::ScalarField],
    w: usize,
) -> Vec<Projective<C>> {
    if p.is_zero() {
        return vec![Projective::zero(); scalars.len()];
    }
    if scalars.len() < 2 {
        return scalars.iter().map(|k| curve_msm(&[*p], &[*k])).collect();
    }
    let halves: Vec<[(bool, Vec<i8>); 2]> = scalars
        .iter()
        .map(|k| {
            let ((s1, k1), (s2, k2)) = C::scalar_decomposition(*k);
            [(s1, k1), (s2, k2)].map(|(positive, half)| (positive, fixed_base_digits(&half.into_bigint(), w)))
        })
        .collect();
    let windows = halves.iter().flatten().map(|h| h.1.len()).max().unwrap_or(0);
    let per = 1usize << (w - 1);
    let mut table = Vec::with_capacity(windows * per);
    let mut base = *p;
    for _ in 0..windows {
        let mut cur = base;
        for _ in 0..per {
            table.push(cur);
            cur += base;
        }
        for _ in 0..w {
            base.double_in_place();
        }
    }
    let table = Projective::normalize_batch(&table);
    let endo: Vec<Affine<C>> = table.iter().map(C::endomorphism_affine).collect();
    halves
        .iter()
        .map(|h| {
            let mut acc = Projective::<C>::zero();
            for ((positive, digits), tab) in h.iter().zip([&table, &endo]) {
                for (j, &d) in digits.iter().enumerate() {
                    if d != 0 {
                        let q = &tab[j * per + d.unsigned_abs() as usize - 1];
                        if (d > 0) == *positive {
                            acc += q;
                        } else {
                            acc -= q;
                        }
                    }
                }
            }
            acc
        })
        .collect()
}

// ------------------------------------------------------------------ GT

/// k = k0 + k1 U + k2 U^2 + k3 U^3 with 0 <= k_i < U (k < r < U^4).
fn base_u_digits(k: &Fr) -> [u64; 4] {
    let mut limbs = k.into_bigint().0;
    let mut out = [0u64; 4];
    for o in out.iter_mut().take(3) {
        let mut rem: u128 = 0;
        for l in limbs.iter_mut().rev() {
            let cur = (rem << 64) | *l as u128;
            *l = (cur / U as u128) as u64;
            rem = cur % U as u128;
        }
        *o = rem as u64;
    }
    debug_assert!(limbs[1..].iter().all(|&l| l == 0) && limbs[0] < U);
    out[3] = limbs[0];
    out
}

/// Odd powers f, f^3, ..., f^(2^(w-1) - 1) of f and of f^U, f^U^2, f^U^3
/// (Frobenius images of the first table), each with their inverses.
struct GtTables {
    pos: [Vec<Fq12>; 4],
    neg: [Vec<Fq12>; 4],
}

fn gt_tables(f: &Fq12) -> GtTables {
    let half = 1 << (GT_W - 2);
    let sq = f.cyclotomic_square();
    let mut base = Vec::with_capacity(half);
    let mut cur = *f;
    for _ in 0..half {
        base.push(cur);
        cur *= sq;
    }
    let conj = |x: &Fq12| x.cyclotomic_inverse().expect("GT elements are nonzero");
    let pos: [Vec<Fq12>; 4] = std::array::from_fn(|i| {
        base.iter()
            .map(|x| match i {
                0 => *x,
                // f^U = (f^u)^-1 = conj(frob(f)), f^U^3 = conj(frob^3(f))
                1 | 3 => conj(&x.frobenius_map(i)),
                _ => x.frobenius_map(i),
            })
            .collect()
    });
    let neg = std::array::from_fn(|i| pos[i].iter().map(conj).collect());
    GtTables { pos, neg }
}

struct GtTerm<'a> {
    digits: Vec<i8>,
    pos: &'a [Fq12],
    neg: &'a [Fq12],
}

fn gt_terms<'a>(k: &Fr, t: &'a GtTables, out: &mut Vec<GtTerm<'a>>) {
    for (i, d) in base_u_digits(k).into_iter().enumerate() {
        if d != 0 {
            let digits = wnaf(&<Fr as PrimeField>::BigInt::from(d), GT_W);
            out.push(GtTerm { digits, pos: &t.pos[i], neg: &t.neg[i] });
        }
    }
}

fn straus_gt(terms: &[GtTerm<'_>]) -> Fq12 {
    let len = terms.iter().map(|t| t.digits.len()).max().unwrap_or(0);
    let mut acc = Fq12::one();
    for j in (0..len).rev() {
        acc.cyclotomic_square_in_place();
        for t in terms {
            match t.digits.get(j) {
                Some(&d) if d != 0 => {
                    let (i, pos) = digit(d);
                    acc *= if pos { &t.pos[i] } else { &t.neg[i] };
                }
                _ => {}
            }
        }
    }
    acc
}

/// prod_i f_i^{k_i} for f_i in GT.
pub fn gt_msm(bases: &[Fq12], scalars: &[Fr]) -> Fq12 {
    let pairs: Vec<_> = bases.iter().zip(scalars).filter(|(f, k)| !f.is_one() && !k.is_zero()).collect();
    let tables: Vec<GtTables> = pairs.iter().map(|(f, _)| gt_tables(f)).collect();
    let mut terms = Vec::with_capacity(4 * pairs.len());
    for ((_, k), t) in pairs.iter().zip(&tables) {
        gt_terms(k, t, &mut terms);
    }
    straus_gt(&terms)
}

/// f^{k_i} for every k_i, f in GT.
pub fn gt_pow_many(f: &Fq12, scalars: &[Fr]) -> Vec<Fq12> {
    let t = gt_tables(f);
    scalars
        .iter()
        .map(|k| {
            let mut terms = Vec::with_capacity(4);
            gt_terms(k, &t, &mut terms);
            straus_gt(&terms)
        })
        .collect()
}

/// Whether f lies in GT, the subgroup of order r of Fq12^*: f is in the
/// cyclotomic subgroup (f^(p^4) f = f^(p^2), as Phi_12(p) = p^4 - p^2 + 1) and
/// f^p = f^u (Scott, ePrint 2021/1130: the cyclotomic subgroup is cyclic and
/// gcd(Phi_12(p), p - u) = r for BLS12-381). About 64 cyclotomic squarings,
/// against a 255-bit exponentiation for f^r = 1.
pub fn gt_is_member(f: &Fq12) -> bool {
    if f.is_zero() {
        return false;
    }
    if f.frobenius_map(4) * f != f.frobenius_map(2) {
        return false;
    }
    // u < 0: f^u = (f^U)^-1, the cyclotomic inverse
    let f_u = f.cyclotomic_exp([U]).cyclotomic_inverse().expect("f is nonzero");
    f.frobenius_map(1) == f_u
}
