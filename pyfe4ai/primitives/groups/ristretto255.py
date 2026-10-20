"""The prime-order group ristretto255 (RFC 9496), reference implementation.

Group elements are equivalence classes of edwards25519 points, represented
here by extended coordinates ``(X, Y, Z, T)`` with ``x = X/Z``, ``y = Y/Z``,
``x y = T/Z`` over ``GF(2^255 - 19)``, and encoded canonically in 32 bytes
(RFC 9496 Sec. 4.3). Scalars live in ``Z_L``, ``L = 2^252 + 2774...493``.
:func:`hash_to_group` is ``hash_to_ristretto255`` of RFC 9380 (suite
``ristretto255_XMD:SHA-512_R255MAP_RO_``): ``expand_message_xmd`` with
SHA-512 to 64 bytes, then the one-way map of RFC 9496 Sec. 4.3.4.

Pure Python on Python integers: a reference, a fallback and a test oracle
for the native backend, about a millisecond per scalar multiplication. It
is **not constant-time** (branches and big-integer arithmetic depend on
secret scalars), like the rest of this research prototype.
"""

from __future__ import annotations

import hashlib
import secrets

#: Field modulus ``2^255 - 19``.
P = 2 ** 255 - 19
#: Order of the ristretto255 group.
L = 2 ** 252 + 27742317777372353535851937790883648493

D = (-121665 * pow(121666, -1, P)) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)
ONE_MINUS_D_SQ = (1 - D * D) % P
D_MINUS_ONE_SQ = (D - 1) * (D - 1) % P
_D2 = 2 * D % P


def _is_negative(x: int) -> bool:
    """RFC 9496 IS_NEGATIVE: the least significant bit of ``x mod p``."""
    return (x % P) & 1 == 1


def _abs(x: int) -> int:
    x %= P
    return P - x if x & 1 else x


def sqrt_ratio_m1(u: int, v: int) -> tuple[bool, int]:
    """RFC 9496 SQRT_RATIO_M1: ``(was_square, r)`` with ``r = sqrt(u/v)`` (non-negative)
    if ``u/v`` is a square, else ``sqrt(i u / v)``."""
    u %= P
    v %= P
    v3 = v * v % P * v % P
    v7 = v3 * v3 % P * v % P
    r = u * v3 % P * pow(u * v7 % P, (P - 5) // 8, P) % P
    check = v * r % P * r % P
    correct = check == u
    flipped = check == (-u) % P
    flipped_i = check == (-u * SQRT_M1) % P
    if flipped or flipped_i:
        r = r * SQRT_M1 % P
    return correct or flipped, _abs(r)


_, _SQRT_AD_MINUS_ONE = sqrt_ratio_m1((-D - 1) % P, 1)  # sqrt(a d - 1), a = -1, sign fixed below
_, INVSQRT_A_MINUS_D = sqrt_ratio_m1(1, (-1 - D) % P)   # 1 / sqrt(a - d)
#: ``sqrt(a d - 1)`` with the sign of RFC 9496 Sec. 4.1 (odd).
SQRT_AD_MINUS_ONE = _SQRT_AD_MINUS_ONE if _is_negative(_SQRT_AD_MINUS_ONE) else P - _SQRT_AD_MINUS_ONE

IDENTITY = (0, 1, 1, 0)


def _base_point():
    y = 4 * pow(5, -1, P) % P
    # x^2 = (y^2 - 1) / (d y^2 + 1), x even (positive)
    was_square, x = sqrt_ratio_m1(y * y - 1, D * y * y + 1)
    assert was_square
    return (x, y, 1, x * y % P)


#: The generator of RFC 9496 (the edwards25519 base point).
BASE = _base_point()


# ------------------------------------------------------------------ arithmetic

def add(p1, p2):
    """Sum of two points (complete twisted-Edwards addition, a = -1)."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = t1 * _D2 % P * t2 % P
    d = 2 * z1 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def neg(p1):
    x, y, z, t = p1
    return ((-x) % P, y, z, (-t) % P)


def sub(p1, p2):
    return add(p1, neg(p2))


def double(p1):
    return add(p1, p1)


def mul(point, k: int):
    """``k * point`` (any integer *k*, reduced mod L), 4-bit fixed window."""
    k %= L
    if k == 0:
        return IDENTITY
    table = [IDENTITY, point]
    for _ in range(14):
        table.append(add(table[-1], point))
    acc = IDENTITY
    for shift in range((k.bit_length() + 3) // 4 * 4 - 4, -1, -4):
        acc = double(double(double(double(acc))))
        nib = (k >> shift) & 15
        if nib:
            acc = add(acc, table[nib])
    return acc


_BASE_COMB: list | None = None


def mul_base(k: int):
    """``k * BASE`` with a precomputed table of ``j 16^i BASE``."""
    global _BASE_COMB
    if _BASE_COMB is None:
        comb, point = [], BASE
        for _ in range(64):
            row = [IDENTITY, point]
            for _ in range(14):
                row.append(add(row[-1], point))
            comb.append(row)
            point = double(double(double(double(point))))
        _BASE_COMB = comb
    k %= L
    acc = IDENTITY
    i = 0
    while k:
        nib = k & 15
        if nib:
            acc = add(acc, _BASE_COMB[i][nib])
        k >>= 4
        i += 1
    return acc


def equal(p1, p2) -> bool:
    """RFC 9496 Sec. 4.5: equality of the classes."""
    x1, y1, _, _ = p1
    x2, y2, _, _ = p2
    return (x1 * y2 - y1 * x2) % P == 0 or (y1 * y2 - x1 * x2) % P == 0


# -------------------------------------------------------------------- encoding

def encode(point) -> bytes:
    """Canonical 32-byte encoding (RFC 9496 Sec. 4.3.2)."""
    x0, y0, z0, t0 = point
    u1 = (z0 + y0) * (z0 - y0) % P
    u2 = x0 * y0 % P
    _, invsqrt = sqrt_ratio_m1(1, u1 * u2 % P * u2 % P)
    den1 = invsqrt * u1 % P
    den2 = invsqrt * u2 % P
    z_inv = den1 * den2 % P * t0 % P
    ix0 = x0 * SQRT_M1 % P
    iy0 = y0 * SQRT_M1 % P
    enchanted = den1 * INVSQRT_A_MINUS_D % P
    if _is_negative(t0 * z_inv):
        x, y, den_inv = iy0, ix0, enchanted
    else:
        x, y, den_inv = x0, y0, den2
    if _is_negative(x * z_inv):
        y = (-y) % P
    s = _abs(den_inv * (z0 - y))
    return s.to_bytes(32, "little")


def decode(data) -> tuple:
    """The point of a canonical encoding (RFC 9496 Sec. 4.3.1).

    Raises:
        ValueError: If *data* is not the canonical encoding of a group element.
    """
    data = bytes(data)
    if len(data) != 32:
        raise ValueError("a ristretto255 encoding has 32 bytes")
    s = int.from_bytes(data, "little")
    if s >= P or _is_negative(s):
        raise ValueError("non-canonical ristretto255 encoding")
    ss = s * s % P
    u1 = (1 - ss) % P
    u2 = (1 + ss) % P
    u2_sqr = u2 * u2 % P
    v = (-(D * u1 % P * u1) - u2_sqr) % P
    was_square, invsqrt = sqrt_ratio_m1(1, v * u2_sqr % P)
    den_x = invsqrt * u2 % P
    den_y = invsqrt * den_x % P * v % P
    x = _abs(2 * s * den_x)
    y = u1 * den_y % P
    t = x * y % P
    if not was_square or _is_negative(t) or y == 0:
        raise ValueError("invalid ristretto255 encoding")
    return (x, y, 1, t)


# -------------------------------------------------------------- hash to group

def _map(t: int):
    """The Elligator-based map of RFC 9496 Sec. 4.3.4 (one field element)."""
    r = SQRT_M1 * t % P * t % P
    u = (r + 1) * ONE_MINUS_D_SQ % P
    v = (-1 - r * D) * (r + D) % P
    was_square, s = sqrt_ratio_m1(u, v)
    s_prime = (-_abs(s * t)) % P
    if not was_square:
        s = s_prime
        c = r
    else:
        c = P - 1
    n = (c * (r - 1) % P * D_MINUS_ONE_SQ - v) % P
    w0 = 2 * s * v % P
    w1 = n * SQRT_AD_MINUS_ONE % P
    w2 = (1 - s * s) % P
    w3 = (1 + s * s) % P
    return (w0 * w3 % P, w2 * w1 % P, w1 * w3 % P, w0 * w2 % P)


def from_uniform_bytes(data) -> tuple:
    """RFC 9496 Sec. 4.3.4: the group element of 64 uniform bytes."""
    data = bytes(data)
    if len(data) != 64:
        raise ValueError("need 64 bytes")
    r0 = int.from_bytes(data[:32], "little") & ((1 << 255) - 1)
    r1 = int.from_bytes(data[32:], "little") & ((1 << 255) - 1)
    return add(_map(r0 % P), _map(r1 % P))


def expand_message_xmd(msg: bytes, dst: bytes, length: int) -> bytes:
    """RFC 9380 Sec. 5.3.1 with SHA-512."""
    b_in, r_in = 64, 128
    if len(dst) > 255 or length > 255 * b_in or length > 65535:
        raise ValueError("expand_message_xmd: DST or length too large")
    ell = -(-length // b_in)
    dst_prime = bytes(dst) + bytes([len(dst)])
    msg_prime = bytes(r_in) + bytes(msg) + length.to_bytes(2, "big") + b"\x00" + dst_prime
    b0 = hashlib.sha512(msg_prime).digest()
    blocks = [hashlib.sha512(b0 + b"\x01" + dst_prime).digest()]
    for i in range(2, ell + 1):
        prev = bytes(a ^ b for a, b in zip(b0, blocks[-1]))
        blocks.append(hashlib.sha512(prev + bytes([i]) + dst_prime).digest())
    return b"".join(blocks)[:length]


def hash_to_group(msg: bytes, dst: bytes) -> tuple:
    """``hash_to_ristretto255`` of RFC 9380 (``ristretto255_XMD:SHA-512_R255MAP_RO_``)."""
    return from_uniform_bytes(expand_message_xmd(msg, dst, 64))


def random_scalar() -> int:
    """A uniform scalar in ``[0, L)`` from the OS CSPRNG."""
    return secrets.randbelow(L)
