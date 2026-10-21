"""Every pairing-based scheme runs over an asymmetric group and rejects symmetric ones."""

import pytest

from pyfe4ai.core import registry
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP
from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup, check_asymmetric, pairing_available

PAIRING_SCHEMES = sorted(
    registry.get_scheme(scheme).key for scheme in registry.list_schemes(capabilities=registry.PAIRING_GROUP)
)
_CONFIG = {
    "n": 2, "m": 2, "s": 2, "t": 2, "eta": 2,
    "lst_nid": ["nid_0", "nid_1"], "lst_sid": ["sid_0", "sid_1"],
    "bound": 3, "bound_x": 3, "bound_y": 3,
}
#: BLS12-381 from the Rust extension; the others from charm-crypto
ASYMMETRIC_GROUPS = ["BLS12-381", "MNT224", "BN254"]


def _need(name):
    if not pairing_available(name):
        pytest.skip("no backend for pairing group {}".format(name))


def _key_generator(key):
    return registry.get_scheme(key).key_generator


def test_registry_has_ten_pairing_schemes():
    assert len(PAIRING_SCHEMES) == 10


@pytest.mark.parametrize("name", ["SS512", "SS1024"])
def test_symmetric_groups_rejected(name):
    _need(name)
    with pytest.raises(FEValidationError, match="symmetric"):
        check_asymmetric(name)
    with pytest.raises(FEValidationError, match="symmetric"):
        check_asymmetric(PairingGroup(name))


@pytest.mark.parametrize("name", ASYMMETRIC_GROUPS)
def test_asymmetric_groups_accepted(name):
    _need(name)
    check_asymmetric(name)


@pytest.mark.parametrize("key", PAIRING_SCHEMES, ids="/".join)
def test_key_generator_rejects_symmetric_group(key):
    _need("SS512")
    with pytest.raises(FEValidationError, match="symmetric"):
        _key_generator(key)(dict(_CONFIG, pairing_group_param="SS512"))


@pytest.mark.parametrize("key", PAIRING_SCHEMES, ids="/".join)
def test_key_generator_defaults_to_asymmetric_group(key):
    _need(DEFAULT_PAIRING_GROUP)
    assert DEFAULT_PAIRING_GROUP == "BLS12-381"
    assert _key_generator(key)(dict(_CONFIG)).pairing_group_param == DEFAULT_PAIRING_GROUP


@pytest.mark.parametrize("name", ASYMMETRIC_GROUPS)
def test_identity_elements_round_trip(name):
    """charm's compressed G1 identity used to deserialise to a non-member point."""
    from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, GT, ZR

    _need(name)
    group = PairingGroup(name)
    for kind in (G1, G2, GT):
        one = group.random(kind) ** group.init(ZR, 0)
        back = group.deserialize(group.serialize(one))
        assert back == one
        if kind == G1:  # charm's ismember rejects the G2 / G_T identities themselves
            assert group.ismember(back)
        x = group.random(kind)
        assert group.deserialize(group.serialize(x)) == x


def test_unknown_group_without_charm_names_the_dependency():
    if pairing_available("MNT224"):
        pytest.skip("charm-crypto is installed")
    with pytest.raises(ImportError, match="charm-crypto"):
        PairingGroup("MNT224")


@pytest.mark.parametrize("key", PAIRING_SCHEMES, ids="/".join)
def test_standard_profile_uses_bls12_381(key):
    from pyfe4ai import FESecurityWarning

    _need("BLS12-381")
    with pytest.warns(FESecurityWarning, match="120 to 126-bit"):
        kg = _key_generator(key)(dict(_CONFIG, security_profile="standard"))
    assert kg.pairing_group_param == "BLS12-381"


def test_standard_profile_fh_ipe_round_trip():
    from pyfe4ai.schemes.sife.fh_ipe_pairing import SIFEFHIPEPairing

    _need("BLS12-381")
    with pytest.warns(Warning):
        kg = _key_generator(("sife", "fh_ipe_pairing"))(dict(_CONFIG, security_profile="standard"))
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFEFHIPEPairing({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": [2, -1]})
    assert SIFEFHIPEPairing({"keys": {"pp": pp}}).decrypt(enc.encrypt([3, 1]), dk, [2, -1]) == 5
