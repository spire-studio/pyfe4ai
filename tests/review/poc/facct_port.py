"""Exact Python port of rust/core/src/facct.rs exp_neg / accept (as reviewed in
October 2026), constants parsed from the Rust source; checks the constants and
the error bound.
"""
import re, sys, random
import gmpy2 as gp

SRC = sys.argv[1] if len(sys.argv) > 1 else None
src = open(SRC).read()
F = 120
ONE = 1 << F
LN2 = int(re.search(r"const LN2: u128 = (0x[0-9a-f]+);", src).group(1), 16)
def arr(name):
    body = re.search(r"const %s: \[u128; \d+\] = \[(.*?)\];" % name, src, re.S).group(1)
    return [int(t, 16) for t in re.findall(r"0x[0-9a-f]+", body)]
EXP_J64 = arr("EXP_J64"); INV_FACT = arr("INV_FACT")
assert len(EXP_J64) == 64 and len(INV_FACT) == 31

with gp.context(precision=600):
    assert LN2 == int(gp.floor(gp.log(2) * ONE)), "LN2"
    bad = [j for j in range(64) if EXP_J64[j] != int(gp.floor(gp.exp(gp.mpfr(-j) / 64) * ONE))]
    print("EXP_J64 wrong entries:", bad)
    import math
    badf = [i for i in range(31) if INV_FACT[i] != ONE // math.factorial(i)]
    print("INV_FACT wrong entries:", badf)

M128 = (1 << 128) - 1
def mul_f(a, b):
    p = a * b
    hi, lo = p >> 128, p & M128
    return ((hi << (128 - F)) | (lo >> F)) & M128  # u128 wrap as in Rust

def shr_pair(hi, lo, s):
    if s == 0:
        return lo if hi == 0 else None
    if 1 <= s <= 127:
        return (((hi << (128 - s)) & M128) | (lo >> s)) if (hi >> s) == 0 else None
    if 128 <= s <= 255:
        return hi >> (s - 128)
    return 0

def exp_neg(c_mant, c_exp, t):
    p = c_mant * t
    hi, lo = p >> 128, p & M128
    s = -(c_exp + F)
    if s < 0:
        return None
    a = shr_pair(hi, lo, s)
    if a is None:
        return None
    if a >= 120 * ONE:
        return 0
    m = int(float(a) / float(LN2))
    while m > 0 and m * LN2 > a:
        m -= 1
    while (m + 1) * LN2 <= a:
        m += 1
    r = a - m * LN2
    j = r >> (F - 6)
    r2 = r - (j << (F - 6))
    pp = INV_FACT[15]
    for i in range(14, -1, -1):
        sub = mul_f(pp, r2)
        assert INV_FACT[i] >= sub, "u128 underflow"
        pp = INV_FACT[i] - sub
    e = mul_f(EXP_J64[j], pp)
    return 0 if m >= 128 else e >> m

P = 192
MARGIN = 1 << (P - 100)

def table_c(neg_c):
    mant, exp = (-neg_c).as_mantissa_exp()
    mant, exp = int(mant), int(exp)
    shift = mant.bit_length() - 128
    mant = mant >> shift if shift > 0 else mant << -shift
    return mant, exp + shift

def decide(c_mant, c_exp, k, x, y, v, exact_T):
    t = y * (y + 2 * k * x)
    e = exp_neg(c_mant, c_exp, t)
    if e is not None:
        approx = e << (P - F)
        if v + 1 + MARGIN <= approx:
            return True, False
        if v >= approx + MARGIN:
            return False, False
    return v < exact_T, True
