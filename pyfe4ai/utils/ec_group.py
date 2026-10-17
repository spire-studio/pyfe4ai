"""Group layer of the elliptic-curve DDH schemes (ristretto255).

The ``*/ddh_ristretto255`` schemes run the DDH constructions of the library
(ABDP15, ACFGU18 Fig. 4, CDGPP18) in the prime-order group ristretto255
(RFC 9496) instead of the order-``q`` subgroup of ``Z_p^*``. This module holds
what they share: the serialisation of group elements (64 hex digits of the
canonical encoding) and scalars (decimal strings mod ``L``), hashing labels
into the group (RFC 9380 ``hash_to_ristretto255``), and the bounded discrete
logarithm of decryption (baby-step giant-step over encodings).

Arithmetic runs in the Rust extension (curve25519-dalek, constant-time scalar
multiplication) when :func:`pyfe4ai.utils.native_backend.ec_ext` provides it,
with group elements held as their 32-byte encodings, and otherwise in
:mod:`pyfe4ai.utils.ristretto255` (pure Python, not constant-time) on
extended coordinates. Every function accepts both forms.
"""

from __future__ import annotations

import functools
import math
import re

from pyfe4ai.utils import ristretto255 as r255
from pyfe4ai.utils.exceptions import FEValidationError

GROUP = "ristretto255"
#: Group order.
ORDER = r255.L
#: Domain-separation tag prefix of the label hashes (RFC 9380 Sec. 3.1).
_DST_SUFFIX = "-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_"

G = r255.BASE
IDENTITY = r255.IDENTITY
#: Largest bound of :func:`dlog` (``ValueError`` above), as the extension's.
MAX_DLOG_BOUND = 2 ** 48
#: A group element as text: exactly 64 hex digits (either case), on both backends.
_HEX64 = re.compile(r"[0-9a-fA-F]{64}")


def _ext():
    from pyfe4ai.utils import native_backend

    return native_backend.ec_ext()


def _b(point) -> bytes:
    return point if isinstance(point, bytes) else r255.encode(point)


def _t(point):
    return r255.decode(point) if isinstance(point, bytes) else point


def _k(k) -> bytes:
    return (int(k) % ORDER).to_bytes(32, "little")


def add(a, b):
    ext = _ext()
    return ext.r255_add(_b(a), _b(b)) if ext else r255.add(_t(a), _t(b))


def sub(a, b):
    ext = _ext()
    return ext.r255_sub(_b(a), _b(b)) if ext else r255.sub(_t(a), _t(b))


def neg(a):
    ext = _ext()
    return ext.r255_neg(_b(a)) if ext else r255.neg(_t(a))


def mul(point, k):
    ext = _ext()
    return ext.r255_mul(_b(point), _k(k)) if ext else r255.mul(_t(point), k)


def mul_base(k):
    ext = _ext()
    return ext.r255_mul_base(_k(k)) if ext else r255.mul_base(k)


def random_scalar() -> int:
    return r255.random_scalar()


def scalar_str(k: int) -> str:
    return str(int(k) % ORDER)


def scalar_from(value, what: str = "scalar") -> int:
    """A scalar from its decimal string (or integer), reduced mod ``L``."""
    try:
        return int(str(value)) % ORDER
    except (TypeError, ValueError) as exc:
        raise FEValidationError("malformed {}".format(what)) from exc


def to_hex(point) -> str:
    return _b(point).hex()


def from_hex(value, what: str = "group element"):
    """The group element of a 64-hex-digit canonical encoding (exactly 64 hex
    digits for a string: no signs, spaces or other separators, as natively).

    Raises:
        FEValidationError: If *value* is not the encoding of a group element.
    """
    try:
        if isinstance(value, str):
            if not _HEX64.fullmatch(value):
                raise ValueError("not 64 hex digits")
            data = bytes.fromhex(value)
        else:
            data = bytes(value)
        ext = _ext()
        if ext:
            if not ext.r255_check(data):
                raise ValueError("invalid encoding")
            return data
        return r255.decode(data)
    except (TypeError, ValueError) as exc:
        raise FEValidationError("malformed {}".format(what)) from exc


def dst(purpose: str) -> bytes:
    """Domain-separation tag ``PYFE4AI-<purpose>-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_``."""
    return ("PYFE4AI-" + purpose + _DST_SUFFIX).encode("ascii")


def hash_to_group(message: str, tag: bytes, index: int = 0):
    """``hash_to_ristretto255(I2OSP(index, 1) || message, tag)`` (RFC 9380)."""
    if not 0 <= index < 256:
        raise ValueError("index must fit one byte")
    msg = bytes([index]) + message.encode("utf-8")
    ext = _ext()
    return ext.r255_hash(msg, tag) if ext else r255.hash_to_group(msg, tag)


# ------------------------------------------------------------- discrete log

#: Baby steps per square root of the range: a larger table (built once per
#: process and bound) means fewer giant steps, each an addition and an
#: encoding (about 0.15 ms in Python).
_BABY_FACTOR = 16


@functools.lru_cache(maxsize=16)
def _dlog_table(bound: int) -> tuple:
    span = 2 * bound + 1
    m = min(span, (math.isqrt(span) + 1) * _BABY_FACTOR)
    table, point = {}, IDENTITY
    for j in range(m):
        table[r255.encode(point)] = j
        point = r255.add(point, G)
    return table, m, r255.mul(G, -m)


def dlog(point, bound: int) -> int | None:
    """The ``x`` with ``x G = point`` and ``|x| <= bound``, or ``None``.

    Raises:
        ValueError: If *bound* is negative or above :data:`MAX_DLOG_BOUND`.
    """
    bound = int(bound)
    if not 0 <= bound <= MAX_DLOG_BOUND:
        raise ValueError("dlog bound must be between 0 and 2^48, got {}".format(bound))
    ext = _ext()
    if ext:
        return ext.r255_dlog(_b(point), bound)
    point = _t(point)
    table, m, giant = _dlog_table(bound)
    gamma = r255.add(point, r255.mul_base(bound))
    for i in range((2 * bound) // m + 2):
        j = table.get(r255.encode(gamma))
        if j is not None:
            x = i * m + j - bound
            return x if abs(x) <= bound else None
        gamma = r255.add(gamma, giant)
    return None
