"""Group layer of the elliptic-curve DDH schemes (ristretto255).

The ``*/ddh_ristretto255`` schemes run the DDH constructions of the library
(ABDP15, ACFGU18 Fig. 4, CDGPP18) in the prime-order group ristretto255
(RFC 9496) instead of the order-``q`` subgroup of ``Z_p^*``. This module holds
what they share: the serialisation of group elements (64 hex digits of the
canonical encoding) and scalars (decimal strings mod ``L``), hashing labels
into the group (RFC 9380 ``hash_to_ristretto255``), and the bounded discrete
logarithm of decryption (baby-step giant-step over encodings). Arithmetic is
:mod:`pyfe4ai.utils.ristretto255`, pure Python and not constant-time.
"""

from __future__ import annotations

import functools
import math

from pyfe4ai.utils import ristretto255 as r255
from pyfe4ai.utils.exceptions import FEValidationError

GROUP = "ristretto255"
#: Group order.
ORDER = r255.L
#: Domain-separation tag prefix of the label hashes (RFC 9380 Sec. 3.1).
_DST_SUFFIX = "-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_"

G = r255.BASE
IDENTITY = r255.IDENTITY

add = r255.add
sub = r255.sub
neg = r255.neg
mul = r255.mul
mul_base = r255.mul_base


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
    return r255.encode(point).hex()


def from_hex(value, what: str = "group element"):
    """The group element of a 64-hex-digit canonical encoding.

    Raises:
        FEValidationError: If *value* is not the encoding of a group element.
    """
    try:
        data = bytes.fromhex(value) if isinstance(value, str) else bytes(value)
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
    return r255.hash_to_group(bytes([index]) + message.encode("utf-8"), tag)


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
        point = add(point, G)
    return table, m, mul(G, -m)


def dlog(point, bound: int) -> int | None:
    """The ``x`` with ``x G = point`` and ``|x| <= bound``, or ``None``."""
    bound = int(bound)
    table, m, giant = _dlog_table(bound)
    gamma = add(point, mul_base(bound))
    for i in range((2 * bound) // m + 2):
        j = table.get(r255.encode(gamma))
        if j is not None:
            x = i * m + j - bound
            return x if abs(x) <= bound else None
        gamma = add(gamma, giant)
    return None
