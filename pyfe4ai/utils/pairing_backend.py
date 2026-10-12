"""Helpers for optional pairing-based dependencies."""

from pyfe4ai.utils.exceptions import FEValidationError

PAIRING_IMPORT_ERROR = None

try:
    from charm.toolbox.pairinggroup import PairingGroup as _CharmPairingGroup
    from charm.toolbox.pairinggroup import ZR, G1, G2, GT, pair
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    _CharmPairingGroup = PairingGroup = ZR = G1 = G2 = GT = pair = None
    PAIRING_IMPORT_ERROR = exc

if _CharmPairingGroup is not None:

    class PairingGroup(_CharmPairingGroup):
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

#: Default pairing group of every pairing-based scheme (asymmetric, type 3).
DEFAULT_PAIRING_GROUP = "MNT224"


def require_pairing_backend() -> None:
    """Raise a clear error if the optional pairing backend is unavailable."""
    if PAIRING_IMPORT_ERROR is not None:
        raise ImportError(
            "Pairing-based schemes require the optional dependency "
            "`charm-crypto-framework`. Install it with "
            "`pip install charm-crypto-framework` or `pip install -e \".[pairing]\"` "
            "and ensure any required system libraries are available."
        ) from PAIRING_IMPORT_ERROR


def check_asymmetric(group) -> None:
    """Reject symmetric (type-1) pairing groups such as SS512.

    Every pairing-based scheme here assumes an asymmetric group (SXDH, or a
    generic asymmetric bilinear group); in a symmetric group ``G1 = G2`` and
    DDH is easy. A group is symmetric iff it pairs two ``G1`` elements.

    Args:
        group: A ``PairingGroup`` or a group name such as ``"MNT224"``.

    Raises:
        FEValidationError: If the group is symmetric.
    """
    require_pairing_backend()
    if isinstance(group, str):
        group = PairingGroup(group)
    try:
        pair(group.random(G1), group.random(G1))
    except ValueError:
        return
    raise FEValidationError(
        "{} is a symmetric pairing group, where SXDH does not hold; use an "
        "asymmetric group such as {} or BN254".format(group.groupType(), DEFAULT_PAIRING_GROUP)
    )
