"""Pairing groups for the pairing-based schemes.

``PairingGroup(name)`` returns BLS12-381 from the Rust extension
(:mod:`pyfe4ai.utils.bls12_381`, arkworks) for ``name == "BLS12-381"``, the
default, and charm-crypto's group of that name otherwise (MNT224, BN254, ...).
charm is optional when the Rust extension is built; it is the fallback for
BLS12-381 only in the sense that every other curve still comes from charm.
``pair`` dispatches on its arguments, and ``ZR``, ``G1``, ``G2``, ``GT`` are
charm's type codes, which both backends share.
"""

from pyfe4ai.utils import bls12_381 as _bls
from pyfe4ai.utils.exceptions import FEValidationError

PAIRING_IMPORT_ERROR = None
ZR, G1, G2, GT = _bls.ZR, _bls.G1, _bls.G2, _bls.GT

try:
    from charm.toolbox.pairinggroup import PairingGroup as _CharmPairingGroup
    from charm.toolbox.pairinggroup import pair as _charm_pair
    from charm.toolbox.pairinggroup import ZR as _CZR, G1 as _CG1, G2 as _CG2, GT as _CGT
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    _CharmPairingGroup = _charm_pair = None
    PAIRING_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on optional dependency
    if (_CZR, _CG1, _CG2, _CGT) != (ZR, G1, G2, GT):
        raise ImportError("unexpected charm group type codes")

#: Default pairing group of every pairing-based scheme (asymmetric, type 3).
DEFAULT_PAIRING_GROUP = _bls.NAME
#: The curves the Rust extension implements.
NATIVE_PAIRING_GROUPS = (_bls.NAME,)

if _CharmPairingGroup is not None:

    class _CharmGroup(_CharmPairingGroup):
        """charm's ``PairingGroup`` with a serialisation fix for the G1 identity.

        charm's compressed encoding of the identity of G1 (on MNT224, BN254,
        ...) deserialises to a point outside the group, so a key or
        ciphertext component that happens to be the identity (e.g. ``g1^0``
        for a zero weight) was silently corrupted. The identity is written as
        ``b"1:identity"`` instead.
        """

        _G1_IDENTITY = b"1:identity"

        def serialize(self, obj, compression=True):
            if getattr(obj, "type", None) == G1 and obj == self.init(G1):
                return self._G1_IDENTITY
            return super().serialize(obj, compression)

        def deserialize(self, obj, compression=True):
            if obj == self._G1_IDENTITY:
                return self.init(G1)
            return super().deserialize(obj, compression)

else:  # pragma: no cover - depends on optional dependency
    _CharmGroup = None


def is_native_group(name) -> bool:
    """Whether the group *name* (or group object) is served by the Rust extension."""
    if isinstance(name, _bls.PairingGroup):
        return True
    return isinstance(name, str) and name.upper().replace("_", "-") in NATIVE_PAIRING_GROUPS


class _PairingGroupMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _bls.PairingGroup) or (_CharmGroup is not None and isinstance(obj, _CharmGroup))


class PairingGroup(metaclass=_PairingGroupMeta):
    """A pairing group by name: BLS12-381 natively, other curves from charm."""

    def __new__(cls, name=DEFAULT_PAIRING_GROUP, *args, **kwargs):
        if is_native_group(name):
            if not _bls.available():
                raise ImportError(
                    "{} needs the Rust extension pyfe4ai._rust_native "
                    "(python tools/build_rust.py); charm curves such as MNT224 "
                    "or BN254 work without it".format(_bls.NAME)
                )
            return _bls.PairingGroup(_bls.NAME)
        require_pairing_backend(name)
        return _CharmGroup(name, *args, **kwargs)


def pair(a, b):
    """The pairing e(a, b) on the backend the elements come from."""
    if _bls.is_element(a) or _bls.is_element(b):
        return _bls.pair(a, b)
    require_pairing_backend()
    return _charm_pair(a, b)


def multi_pair(lhs, rhs):
    """prod_i e(lhs[i], rhs[i]); one final exponentiation on BLS12-381."""
    lhs, rhs = list(lhs), list(rhs)
    if len(lhs) != len(rhs):
        raise ValueError("multi_pair needs as many G1 as G2 elements")
    if lhs and (_bls.is_element(lhs[0]) or _bls.is_element(rhs[0])):
        return _bls.multi_pair(lhs, rhs)
    out = None
    for a, b in zip(lhs, rhs):
        e = pair(a, b)
        out = e if out is None else out * e
    return out


def require_pairing_backend(name=None) -> None:
    """Raise a clear error if no backend implements the pairing group *name*.

    Without *name*, only checks that some pairing backend is present (the
    Rust extension or charm); :func:`PairingGroup` checks the group itself.
    """
    if name is None and (_bls.available() or PAIRING_IMPORT_ERROR is None):
        return
    if name is not None and is_native_group(name):
        if _bls.available():
            return
        raise ImportError(
            "{} needs the Rust extension pyfe4ai._rust_native (python tools/build_rust.py)".format(_bls.NAME)
        )
    if PAIRING_IMPORT_ERROR is not None:
        raise ImportError(
            "Pairing groups other than {} require the optional dependency "
            "`charm-crypto-framework`. Install it with "
            "`pip install charm-crypto-framework` or `pip install -e \".[pairing]\"` "
            "and ensure any required system libraries are available.".format(_bls.NAME)
        ) from PAIRING_IMPORT_ERROR


def pairing_available(name=DEFAULT_PAIRING_GROUP) -> bool:
    """Whether :func:`PairingGroup` can build the group *name*."""
    try:
        require_pairing_backend(name)
    except ImportError:
        return False
    return True


def check_asymmetric(group) -> None:
    """Reject symmetric (type-1) pairing groups such as SS512.

    Every pairing-based scheme here assumes an asymmetric group (SXDH, or a
    generic asymmetric bilinear group); in a symmetric group ``G1 = G2`` and
    DDH is easy. A group is symmetric iff it pairs two ``G1`` elements.

    Args:
        group: A ``PairingGroup`` or a group name such as ``"BLS12-381"``.

    Raises:
        FEValidationError: If the group is symmetric.
    """
    if isinstance(group, str):
        require_pairing_backend(group)
        group = PairingGroup(group)
    try:
        pair(group.random(G1), group.random(G1))
    except ValueError:
        return
    raise FEValidationError(
        "{} is a symmetric pairing group, where SXDH does not hold; use an "
        "asymmetric group such as {} or BN254".format(group.groupType(), DEFAULT_PAIRING_GROUP)
    )
