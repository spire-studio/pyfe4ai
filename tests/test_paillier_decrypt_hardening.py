"""Paillier IPFE decryption must not act as an oracle on crafted ciphertexts.

With ``ct0 = 1 + N`` and ``ct1 = (1, ..., 1)`` the decryption equation gives
``(1 + N)^{-k}``, i.e. ``-k mod N`` -- the functional key -- unless the result
is checked against the inner-product bound."""

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mife.paillier import MIFEPaillier, MIFEPaillierKeyGenerator
from pyfe4ai.schemes.sife.paillier import SIFEPaillier, SIFEPaillierKeyGenerator
from pyfe4ai.core.exceptions import FEValidationError


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _sife():
    kg = SIFEPaillierKeyGenerator({"eta": 3, "bit_length": 64, "bound_x": 10, "bound_y": 10})
    kg.setup()
    pp = kg.get_public_parameters()
    y = [3, -2, 5]
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    enc = SIFEPaillier({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    return pp, y, dk, enc, SIFEPaillier({"keys": {"pp": pp}})


def test_sife_crafted_ciphertext_does_not_reveal_the_key():
    pp, y, dk, _, dec = _sife()
    n = gp.mpz(pp["n"])
    crafted = {"ct0": gp.digits(n + 1), "ct1": ["1", "1", "1"]}
    assert dec.decrypt(crafted, dk, y) is None


def test_sife_malformed_ciphertexts_are_rejected():
    pp, y, dk, enc, dec = _sife()
    ct = enc.encrypt([1, 2, 3])
    with pytest.raises(FEValidationError):
        dec.decrypt({"ct0": ct["ct0"], "ct1": ct["ct1"][:2]}, dk, y)
    with pytest.raises(FEValidationError):
        dec.decrypt({"ct0": "0", "ct1": ct["ct1"]}, dk, y)
    assert dec.decrypt(ct, dk, y) == 3 * 1 - 2 * 2 + 5 * 3


def test_mife_crafted_ciphertext_does_not_reveal_the_key():
    nids = ["nid_0", "nid_1"]
    kg = MIFEPaillierKeyGenerator({"n": 2, "lst_nid": nids, "eta": 2, "bit_length": 64,
                                   "bound_x": 10, "bound_y": 10})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {"nid_0": [1, 2], "nid_1": [-3, 4]}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y})
    n = gp.mpz(pp["n_mod"])
    crafted = {nid: {"ct0": gp.digits(n + 1), "ct1": ["1", "1"]} for nid in nids}
    dec = MIFEPaillier({"keys": {"pp": pp}})
    assert dec.decrypt(crafted, dk, y) is None
    cts = {nid: MIFEPaillier({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([1, 1])
           for nid in nids}
    assert dec.decrypt(cts, dk, y) == 1 + 2 - 3 + 4
