"""Regressions from the ABDP15 review of SIFE-LWE."""

import numpy as np
import pytest

from pyfe4ai.schemes.sife.lwe import SIFELWE, SIFELWEKeyGenerator
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError

CFG = {"sec_param": 64, "eta": 4, "lwe_n": 16, "bound_x": 8, "bound_y": 8}


@pytest.fixture
def scheme():
    kg = SIFELWEKeyGenerator(CFG)
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, SIFELWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})


@pytest.mark.parametrize("sign", [1, -1])
def test_extreme_inner_products(scheme, sign):
    kg, enc = scheme
    y = [sign * 8] * 4
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    for _ in range(20):
        assert enc.decrypt(enc.encrypt([8] * 4), dk, y) == sign * 256


def test_weights_bound_to_key_and_inputs_validated(scheme):
    kg, enc = scheme
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": [1, 2, 3, 4]})
    ct = enc.encrypt([1, 1, 1, 1])
    with pytest.raises(FESchemeError):
        enc.decrypt(ct, dk, [1, 2, 3, 5])
    with pytest.raises(FEValidationError):
        enc.decrypt({"ct0": ct["ct0"][:-1], "ct1": ct["ct1"]}, dk, [1, 2, 3, 4])
    with pytest.raises(FEValidationError):
        enc.encrypt([1.5, 0, 0, 0])
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(None, credentials={"fusion_weight": [0.5, 0, 0, 0]})


def test_ndarray_helper_checks_bound_x():
    cfg = dict(CFG, eta=1, bound_x=20)
    kg = SIFELWEKeyGenerator(cfg)
    kg.setup()
    enc = SIFELWE({"keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys()}})
    with pytest.raises(FEValidationError):
        enc.encrypt_lst_ndarray([np.array([608.0])])
