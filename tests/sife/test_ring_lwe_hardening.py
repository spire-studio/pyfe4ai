"""Regressions from the BMMS21 review of SIFE-Ring-LWE."""

import json
import os

import pytest

from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE, SIFERingLWEKeyGenerator
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.ring_lwe_utils import DEFAULT_RING_N

CFG = {"sec_param": 64, "eta": 3, "bound_x": 4, "bound_y": 4}


@pytest.fixture(scope="module")
def scheme():
    kg = SIFERingLWEKeyGenerator(CFG)
    kg.setup()
    return kg, SIFERingLWE({"keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys()}})


def test_default_ring_n_is_explicit_and_checked():
    assert SIFERingLWEKeyGenerator(CFG).ring_n == DEFAULT_RING_N
    for bad in (24, 3, 1):
        with pytest.raises(FEValidationError):
            SIFERingLWEKeyGenerator(dict(CFG, ring_n=bad))


def test_extremes_decode_in_every_slot(scheme):
    kg, enc = scheme
    y = [4, -4, 4]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    ct = enc.encrypt([[4, -4], [-4, 4], [4, -4]])
    assert enc.decrypt(ct, dk, y) == [48, -48]


def test_inputs_validated_and_key_bound(scheme):
    kg, enc = scheme
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": [1, 2, 3]})
    with pytest.raises(FEValidationError):
        enc.encrypt([[2.9], [1], [1]])
    with pytest.raises(FEValidationError):
        enc.encrypt([[1], [1, 2, 3], [1]])
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(None, credentials={"fusion_weight": [1.7, 0, 0]})
    ct = enc.encrypt([[1], [2], [3]])
    assert enc.decrypt(ct, dk, [1, 2, 3]) == [14]
    with pytest.raises(FESchemeError):
        enc.decrypt(ct, dk, [1, 2, 4])


def test_mife_parameters_do_not_overwrite_sife_cache(scheme):
    path = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE_RING_LWE, "param.json")
    with open(path, encoding="utf-8") as f:
        before = json.load(f)
    MIFERingLWEKeyGenerator({"sec_param": 64, "n": 2, "eta": 5, "bound_x": 3, "bound_y": 3})
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == before
