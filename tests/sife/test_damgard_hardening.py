"""Regressions from the ALS16 review of SIFE-Damgard."""

import pytest

from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgard, SIFEDamgardKeyGenerator
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

CFG = {"sec_param": 128, "eta": 3, "bound": 20}


@pytest.fixture(scope="module")
def scheme():
    kg = SIFEDamgardKeyGenerator(CFG)
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, pp, SIFEDamgard({"keys": {"pp": pp, "sk": kg.get_private_keys()}})


def test_key_holder_with_sk_decrypts_extremes(scheme):
    kg, pp, enc = scheme
    y = [20, -20, 20]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    ct = enc.encrypt([20, -20, 20])
    assert enc.decrypt(ct, dk, y) == 1200
    assert SIFEDamgard({"keys": {"pp": pp}}).decrypt(ct, dk, y) == 1200


def test_key_binding_and_ciphertext_checks(scheme):
    kg, _, enc = scheme
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": [1, 2, 3]})
    ct = enc.encrypt([1, 1, 1])
    with pytest.raises(FESchemeError):
        enc.decrypt(ct, dk, [1, 2, 4])
    with pytest.raises(FEValidationError):
        enc.decrypt(dict(ct, e=ct["e"][:2]), dk, [1, 2, 3])
    with pytest.raises(FEValidationError):
        enc.encrypt([1.5, 1, 1])


def test_bound_too_large_for_group():
    with pytest.raises(FEValidationError):
        SIFEDamgardKeyGenerator({"sec_param": 32, "eta": 3, "bound": 2 ** 20})
