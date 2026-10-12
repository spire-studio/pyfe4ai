"""Regressions from the ABDP15 review of SIFE-DDH (and its variable-length variant)."""

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.sife.ddh import SIFE, SIFEKeyGenerator
from pyfe4ai.schemes.sife.ddh_dynamic import SIFEDynamic, SIFEDynamicKeyGenerator
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

ETA = 4


@pytest.fixture(params=[(SIFEKeyGenerator, SIFE), (SIFEDynamicKeyGenerator, SIFEDynamic)])
def scheme(request):
    kg_cls, crypto_cls = request.param
    kg = kg_cls({"sec_param": 128, "eta": ETA})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = crypto_cls({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    return kg, pp, enc, crypto_cls


def test_ciphertext_is_reduced_and_hides_x(scheme):
    """Versions <= 0.2.0 published (h^r mod p)(g^x mod p): g^x mod p divided it."""
    kg, pp, enc, _ = scheme
    p, g, q = gp.mpz(pp["p"]), gp.mpz(pp["g"]), kg.q
    x = [17, -5, 3, 250]
    ct = enc.encrypt(x)
    for c, x_i in zip(ct["ct1"], x):
        c = gp.mpz(c)
        assert 0 < c < p and gp.powmod(c, q, p) == 1
        assert c % gp.powmod(g, x_i, p) != 0


def test_key_is_reduced_mod_q(scheme):
    kg, _, _, _ = scheme
    q = int(kg.q)
    k1 = kg.get_decryption_keys(None, credentials={"fusion_weight": [1, q - 1, 0, 0]})
    k2 = kg.get_decryption_keys(None, credentials={"fusion_weight": [1, -1, 0, 0]})
    assert k1["z"] == k2["z"] and 0 <= int(k1["z"]) < q


def test_signed_round_trip_and_key_holder_with_sk_decrypts(scheme):
    kg, pp, enc, crypto_cls = scheme
    y = [2, -3, 1, 0]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    ct = enc.encrypt([5, 4, -6, 9])
    assert enc.decrypt(ct, dk, y) == 10 - 12 - 6
    assert crypto_cls({"keys": {"pp": pp}}).decrypt(ct, dk, y) == -8


def test_inputs_are_validated(scheme):
    kg, _, enc, _ = scheme
    with pytest.raises(FEValidationError):
        enc.encrypt([2.9, 1, 1, 1])
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(None, credentials={"fusion_weight": [1.5, 1, 1, 1]})
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": [1, 1, 1, 1]})
    ct = enc.encrypt([1, 2, 3, 4])
    with pytest.raises(FESchemeError):
        enc.decrypt(ct, dk, [1, 1, 1, 2])
    with pytest.raises(FEValidationError):
        enc.decrypt({"ct0": ct["ct0"], "ct1": ct["ct1"][:2]}, dk, [1, 1, 1, 1])


def test_full_length_required_for_fixed_scheme():
    kg = SIFEKeyGenerator({"sec_param": 128, "eta": ETA})
    kg.setup()
    enc = SIFE({"keys": {"pp": kg.get_public_parameters(), "sk": kg.get_private_keys()}})
    with pytest.raises(FEValidationError):
        enc.encrypt([1, 2])
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(None, credentials={"fusion_weight": [1, 2]})


def test_dlog_bound_option():
    kg = SIFEKeyGenerator({"sec_param": 128, "eta": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": [100, 100]})
    ct = enc.encrypt([30, 20])
    assert SIFE({"keys": {"pp": pp}}).decrypt(ct, dk, [100, 100]) is None
    assert SIFE({"keys": {"pp": pp}, "dlog_bound": 10 ** 4}).decrypt(ct, dk, [100, 100]) == 5000
