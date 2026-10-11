"""Regressions from the ALS16 review of SIFE-FullySec-LWE."""

import json
import os

import pytest

from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWE, SIFEFullySecLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

CFG = {"sec_param": 64, "eta": 2, "lwe_n": 8, "bound_x": 10, "bound_y": 10}


@pytest.fixture(scope="module")
def scheme():
    kg = SIFEFullySecLWEKeyGenerator(CFG)
    kg.setup()
    return kg, SIFEFullySecLWE({"keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys()}})


@pytest.mark.parametrize("sign", [1, -1])
def test_extreme_inner_products(scheme, sign):
    """K = 2 eta X Y made +-eta X Y decode with a random sign about half the time."""
    kg, enc = scheme
    y = [sign * 10, sign * 10]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    assert int(kg.k) == 2 * 2 * 10 * 10 + 1
    for _ in range(30):
        assert enc.decrypt(enc.encrypt([10, 10]), dk, y) == sign * 200
        assert enc.decrypt(enc.encrypt([-10, -10]), dk, y) == -sign * 200


def test_key_binding_and_validation(scheme):
    kg, enc = scheme
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": [3, -4]})
    ct = enc.encrypt([5, 6])
    assert enc.decrypt(ct, dk, [3, -4]) == -9
    with pytest.raises(FESchemeError):
        enc.decrypt(ct, dk, [3, -3])
    with pytest.raises(FEValidationError):
        enc.decrypt({"ct0": ct["ct0"][:-1], "ct1": ct["ct1"]}, dk, [3, -4])
    with pytest.raises(FEValidationError):
        enc.encrypt([5.5, 6])


def test_stale_cache_is_regenerated(scheme):
    kg, _ = scheme
    path = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE_FULLYSEC_LWE, "param.json")
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["construction"] == "als16-lwe/v4"
