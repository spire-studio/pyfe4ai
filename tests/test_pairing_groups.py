"""Every pairing-based scheme runs over an asymmetric group and rejects symmetric ones."""

import importlib

import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.utils.exceptions import FEValidationError  # noqa: E402
from pyfe4ai.utils.ml_adapter import _SCHEME_REGISTRY  # noqa: E402
from pyfe4ai.utils.pairing_backend import DEFAULT_PAIRING_GROUP  # noqa: E402
from pyfe4ai.utils.pairing_backend import PairingGroup, check_asymmetric  # noqa: E402

PAIRING_SCHEMES = sorted(
    key for key in _SCHEME_REGISTRY if "pairing" in key[1] or key[0] == "quadratic"
)
_CONFIG = {
    "n": 2, "m": 2, "s": 2, "t": 2, "eta": 1,
    "lst_nid": ["nid_0", "nid_1"], "lst_sid": ["sid_0", "sid_1"],
    "bound": 3, "bound_x": 3, "bound_y": 3,
}


def _key_generator(key):
    module, kg_name, _ = _SCHEME_REGISTRY[key]
    return getattr(importlib.import_module(module), kg_name)


def test_registry_has_ten_pairing_schemes():
    assert len(PAIRING_SCHEMES) == 10


@pytest.mark.parametrize("name", ["SS512", "SS1024"])
def test_symmetric_groups_rejected(name):
    with pytest.raises(FEValidationError, match="symmetric"):
        check_asymmetric(name)
    with pytest.raises(FEValidationError, match="symmetric"):
        check_asymmetric(PairingGroup(name))


@pytest.mark.parametrize("name", ["MNT224", "BN254"])
def test_asymmetric_groups_accepted(name):
    check_asymmetric(name)


@pytest.mark.parametrize("key", PAIRING_SCHEMES, ids="/".join)
def test_key_generator_rejects_symmetric_group(key):
    with pytest.raises(FEValidationError, match="symmetric"):
        _key_generator(key)(dict(_CONFIG, pairing_group_param="SS512"))


@pytest.mark.parametrize("key", PAIRING_SCHEMES, ids="/".join)
def test_key_generator_defaults_to_asymmetric_group(key):
    assert _key_generator(key)(dict(_CONFIG)).pairing_group_param == DEFAULT_PAIRING_GROUP
