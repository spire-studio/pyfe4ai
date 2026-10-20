"""Conditions an application should see without configuring logging.

The ``pyfe4ai`` logger has a ``NullHandler``: what the library only logged
(a regenerated parameter record, a request answered with ``None``, a slower
fallback) was invisible unless the application configured logging. Each is
also issued as an :class:`~pyfe4ai.core.exceptions.FEWarning`, which is a
``UserWarning`` but not an ``FESecurityWarning`` (silencing one leaves the
other).
"""

import json
import os
import warnings

import pytest

from pyfe4ai.core.base import cache_dir
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FESecurityWarning, FEWarning
from pyfe4ai.schemes.sife.ddh import SIFE, SIFEKeyGenerator

DDH = {"sec_param": 128, "eta": 2}
NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _param_file(scheme_type):
    return os.path.join(cache_dir(), "authority", scheme_type, "param.json")


def _no_fe_warning(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    assert not [w for w in caught if issubclass(w.category, FEWarning)], [str(w.message) for w in caught]
    return out


def test_fe_warning_is_a_user_warning_apart_from_the_security_warning():
    assert issubclass(FEWarning, UserWarning)
    assert not issubclass(FEWarning, FESecurityWarning) and not issubclass(FESecurityWarning, FEWarning)


def test_a_record_that_no_longer_verifies_is_regenerated_with_a_warning():
    SIFEKeyGenerator(dict(DDH))
    _no_fe_warning(lambda: SIFEKeyGenerator(dict(DDH)))  # a matching record: no warning
    path = _param_file(CryptoCONST.TYPE_SIFE)
    with open(path) as f:
        record = json.load(f)
    record["group"] = {"p": "23", "q": "11", "r": "2"}
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.warns(FEWarning, match="regenerated"):
        kg = SIFEKeyGenerator(dict(DDH))
    assert int(kg.p).bit_length() == 128


def test_an_unreadable_record_is_regenerated_with_a_warning():
    SIFEKeyGenerator(dict(DDH))
    with open(_param_file(CryptoCONST.TYPE_SIFE), "w") as f:
        f.write('{"sec_param": 128, "group": ')  # truncated write
    with pytest.warns(FEWarning, match="unreadable.*regenerated"):
        SIFEKeyGenerator(dict(DDH))


def test_a_decryption_out_of_range_warns():
    kg = SIFEKeyGenerator(dict(DDH))
    kg.setup()
    pp = kg.get_public_parameters()
    ct = SIFE({"id": "c", "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys()}}).encrypt([5, 5])
    server = SIFE({"id": "s", "precision": 0, "dlog_bound": 10, "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [5, 5]})
    with pytest.warns(FEWarning, match="out of bound"):
        assert server.decrypt(ct, dk, [5, 5]) is None


def _ring_threshold(**extra):
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(
        n=3, lst_nid=NIDS, s=3, t=3, lst_sid=SIDS, eta=1, ring_n=32, sec_param=64, bound_x=100, bound_y=3,
        stat_param=64, **extra))
    kg.setup()
    return tm, kg, kg.get_public_parameters()


def test_an_unknown_server_id_warns():
    _, kg, _ = _ring_threshold()
    creds = {"fusion_weight": {n: [1] for n in NIDS}, "label": "L"}
    with pytest.warns(FEWarning, match="invalid sid"):
        assert kg.get_decryption_keys("s9", credentials=creds) is None
    with pytest.warns(FEWarning, match="invalid sid"):
        assert kg.get_weight_key_share("s9", credentials=creds) is None


def test_partial_decryptions_that_disagree_warn():
    tm, kg, pp = _ring_threshold(key_shares="full")
    weights = {n: [1] for n in NIDS}
    creds = {"fusion_weight": weights, "label": "L"}
    cts = {n: tm.ThresholdMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
           .encrypt([[1, 2, 3]], "L") for n in NIDS}
    parts = {s: tm.ThresholdMCFERingLWE({"id": s, "precision": 0, "keys": {"pp": pp}}).share_decrypt(
        cts, creds, kg.get_decryption_keys(s, credentials=creds), SIDS) for s in SIDS}
    combiner = tm.ThresholdMCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}})
    assert _no_fe_warning(lambda: combiner.combine_decrypt(parts)) == [3, 6, 9]
    parts["s1"] = dict(parts["s1"], ct_digest="0" * 64)  # decrypted other ciphertexts
    with pytest.warns(FEWarning, match="disagree on the ciphertexts"):
        assert combiner.combine_decrypt(parts) is None


def test_an_rns_modulus_beyond_the_native_ring_warns(monkeypatch):
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    monkeypatch.setattr(tm, "RNS_NATIVE_FACTORS", 2)  # every RNS q of the scheme has three or more factors
    with pytest.warns(FEWarning, match="Python path"):
        _ring_threshold()


def test_a_modulus_without_an_ntt_root_warns():
    from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_mul

    with pytest.warns(FEWarning, match="NTT root not found"):
        assert poly_mul([1, 0, 0, 0], [0, 1, 0, 0], 11) == [0, 1, 0, 0]  # 11 = 3 mod 8: no 8th root of unity
