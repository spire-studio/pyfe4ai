"""Helpers for optional pairing-based dependencies."""

PAIRING_IMPORT_ERROR = None

try:
    from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    PairingGroup = ZR = G1 = G2 = GT = pair = None
    PAIRING_IMPORT_ERROR = exc


def require_pairing_backend() -> None:
    """Raise a clear error if the optional pairing backend is unavailable."""
    if PAIRING_IMPORT_ERROR is not None:
        raise ImportError(
            "Pairing-based schemes require the optional dependency "
            "`charm-crypto-framework`. Install it with "
            "`pip install charm-crypto-framework` or `pip install -e \".[pairing]\"` "
            "and ensure any required system libraries are available."
        ) from PAIRING_IMPORT_ERROR
