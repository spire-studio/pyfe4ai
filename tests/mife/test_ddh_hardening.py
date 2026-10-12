"""Regressions from the review of MIFE-DDH (ACFGU18 Fig. 4 + ABDP15)."""

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mife.ddh import MIFE, MIFEKeyGenerator
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

NIDS = ["nid_0", "nid_1"]


@pytest.fixture(scope="module")
def scheme():
    kg = MIFEKeyGenerator({"sec_param": 128, "lst_nid": NIDS, "eta": 2, "n": 2, "s": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: MIFE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    return kg, pp, clients


def test_ciphertext_reduced_and_not_factor_revealing(scheme):
    """v0.2.0 stored (g^{x+u} mod p) * (mask mod p): factoring revealed x + u."""
    kg, pp, clients = scheme
    p, q = gp.mpz(pp["p"]), gp.mpz(pp["q"])
    ct = clients["nid_0"].encrypt([7, -3])
    for c in ct["c"] + [ct["t"]]:
        c = gp.mpz(c)
        assert 0 < c < p and gp.powmod(c, q, p) == 1


def test_large_weight_key_does_not_reveal_pads(scheme):
    """z and d are reduced mod q: a weight 2^256 no longer exposes u over Z."""
    kg, pp, _ = scheme
    q = gp.mpz(pp["q"])
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": {"nid_0": [2 ** 256, 0], "nid_1": [1, 0]}})
    assert 0 <= gp.mpz(dk["z"]) < q
    assert all(0 <= gp.mpz(v) < q for v in dk["d"].values())


def test_slot_coverage_and_weight_binding(scheme):
    kg, pp, clients = scheme
    y = {"nid_0": [1, 2], "nid_1": [-1, 3]}
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    cts = {nid: c.encrypt([4, 5]) for nid, c in clients.items()}
    dec = MIFE({"keys": {"pp": pp}})
    assert dec.decrypt(cts, dk, y) == 14 + 11
    with pytest.raises(FESchemeError):
        dec.decrypt({"nid_0": cts["nid_0"]}, dk, y)
    with pytest.raises(FESchemeError):
        dec.decrypt(cts, dk, {"nid_0": [1, 2], "nid_1": [-1, 4]})
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys(None, credentials={"fusion_weight": {"nid_0": [1, 2]}})
    with pytest.raises(FEValidationError):
        clients["nid_0"].encrypt([1.5, 2])


def test_dlog_bound_option(scheme):
    kg, pp, clients = scheme
    y = {"nid_0": [600, 0], "nid_1": [400, 0]}
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    cts = {nid: c.encrypt([9, 0]) for nid, c in clients.items()}
    assert MIFE({"keys": {"pp": pp}}).decrypt(cts, dk, y) is None
    assert MIFE({"keys": {"pp": pp}, "dlog_bound": 10 ** 4}).decrypt(cts, dk, y) == 9000
