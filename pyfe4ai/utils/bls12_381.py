"""The pairing group BLS12-381 on the Rust backend, with charm's interface.

:class:`PairingGroup` and :func:`pair` behave as charm-crypto's for the
operations the pairing-based schemes use, so the schemes run unchanged on
BLS12-381 (arkworks, in :mod:`pyfe4ai._rust_native`): ``random(ZR | G1 | G2 |
GT)``, ``init(type, value)``, ``order()``, ``serialize`` / ``deserialize``
(``b"<type>:<base64 of the compressed encoding>"``; deserialisation checks
the encoding and subgroup membership), ``hash(msg, G1 | G2)`` (RFC 9380
hash_to_curve, suites BLS12381G1/G2_XMD:SHA-256_SSWU_RO_, with the domain
separation tags of :data:`HASH_DST`), ``ismember``, ``groupType``; elements
written multiplicatively (``*``, ``/``, ``**``), ZR elements with ``+ - * /
**`` and ``int()``. Products of pairings (:func:`multi_pair`) take one final
exponentiation; :func:`gt_dlog` is a bounded discrete logarithm in GT with a
baby-step table built once per base and bound.

BLS12-381 gives about 120-126 bits of security after the exTNFS attacks on
its 381-bit field (Barbulescu and Duquesne, J. Cryptol. 2019), below the
128-bit target of the ``standard`` profile, which therefore emits a
:class:`~pyfe4ai.utils.exceptions.FESecurityWarning`. Not constant-time.
"""

from __future__ import annotations

import base64

ZR, G1, G2, GT = 0, 1, 2, 3
NAME = "BLS12-381"
#: RFC 9380 domain separation tags of :meth:`PairingGroup.hash`.
HASH_DST = {
    G1: b"PYFE4AI-V01-CS01-with-BLS12381G1_XMD:SHA-256_SSWU_RO_",
    G2: b"PYFE4AI-V01-CS01-with-BLS12381G2_XMD:SHA-256_SSWU_RO_",
}

try:  # the Rust extension, if built
    from pyfe4ai import _rust_native as _ext

    if not hasattr(_ext, "bls_pair"):
        _ext = None
except ImportError:  # pragma: no cover - depends on the build
    _ext = None


def available() -> bool:
    return _ext is not None


def _require():
    if _ext is None:
        raise ImportError(
            "BLS12-381 needs the Rust extension pyfe4ai._rust_native (python tools/build_rust.py)"
        )
    return _ext


class PairingGroup:
    """BLS12-381 with the methods of charm's ``PairingGroup`` that the schemes use."""

    def __init__(self, name: str = NAME, *args, **kwargs) -> None:
        if name.upper().replace("_", "-") != NAME:
            raise ValueError("this backend implements {} only, got {!r}".format(NAME, name))
        self._ext = _require()
        self.param = NAME

    def groupType(self) -> str:
        return NAME

    def order(self) -> int:
        return self._ext.bls_order()

    def random(self, kind=ZR, count: int = 1, seed=None):
        if seed is not None:
            raise ValueError("seeded random elements are not supported")
        if count != 1:
            return [self._ext.bls_random(kind) for _ in range(count)]
        return self._ext.bls_random(kind)

    def init(self, kind, value=None):
        """ZR(value), or the identity of G1 / G2 / GT (charm's ``init(GT, 1)``)."""
        if kind == ZR:
            return self._ext.bls_init(ZR, 0 if value is None else value)
        if value not in (None, 0, 1):
            raise ValueError("group elements are initialised to the identity only")
        return self._ext.bls_init(kind)

    def generator(self, kind):
        """The fixed generator of G1 or G2, or e(g1, g2) for GT."""
        return self._ext.bls_generator(kind)

    def serialize(self, element, compression: bool = True) -> bytes:
        return "{}:".format(element.type).encode("ascii") + base64.b64encode(element.to_bytes())

    def deserialize(self, data, compression: bool = True):
        try:
            if isinstance(data, str):
                data = data.encode("ascii")
            kind, _, body = bytes(data).partition(b":")
            return self._ext.bls_deserialize(int(kind), base64.b64decode(body, validate=True))
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid serialised BLS12-381 element") from exc

    def ismember(self, element) -> bool:
        return getattr(element, "type", None) in (ZR, G1, G2, GT)

    def hash(self, msg, kind=G1):
        """H(msg) in G1 or G2; *msg* is bytes or a str (hashed as UTF-8)."""
        if kind not in HASH_DST:
            raise ValueError("BLS12-381 hashes into G1 or G2 only")
        if isinstance(msg, str):
            msg = msg.encode("utf-8")
        return self._ext.bls_hash(kind, bytes(msg), HASH_DST[kind])


def pair(a, b):
    """e(a, b) for a in G1 and b in G2."""
    return _require().bls_pair(a, b)


def multi_pair(lhs: list, rhs: list):
    """prod_i e(lhs[i], rhs[i]) with one final exponentiation."""
    return _require().bls_multi_pair(list(lhs), list(rhs))


def gt_dlog(base, target, bound: int):
    """The x with base^x = target and |x| <= bound, or None."""
    return _require().bls_gt_dlog(base, target, int(bound))


def is_element(obj) -> bool:
    """Whether *obj* is an element of this backend (not of charm)."""
    return _ext is not None and isinstance(obj, _ext.Bls12381Element)
