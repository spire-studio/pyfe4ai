"""The parameter cache in ./config/authority: a corrupted file is regenerated,
and cached discrete-log and DCR parameters are checked structurally before
they are used (a planted 5-bit group or a prime "RSA modulus" used to be
accepted)."""

import json
import os

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgardKeyGenerator
from pyfe4ai.schemes.mcfe.paillier import MCFEPaillierKeyGenerator
from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgardKeyGenerator
from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator
from pyfe4ai.schemes.sife.paillier import SIFEPaillierKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _param_file(scheme_type):
    return os.path.join("config", "authority", scheme_type, "param.json")


def _rewrite(scheme_type, change):
    path = _param_file(scheme_type)
    with open(path) as f:
        record = json.load(f)
    change(record)
    with open(path, "w") as f:
        json.dump(record, f)


DDH = {"sec_param": 128, "eta": 2}


def test_corrupted_cache_is_regenerated():
    SIFEKeyGenerator(dict(DDH))
    with open(_param_file(CryptoCONST.TYPE_SIFE), "w") as f:
        f.write('{"sec_param": 128, "group": ')  # truncated write
    kg = SIFEKeyGenerator(dict(DDH))
    assert int(kg.p).bit_length() == 128
    with open(_param_file(CryptoCONST.TYPE_SIFE)) as f:
        assert json.load(f)["sec_param"] == 128


def test_cache_writes_leave_no_temporary_files():
    SIFEKeyGenerator(dict(DDH))
    assert os.listdir(os.path.dirname(_param_file(CryptoCONST.TYPE_SIFE))) == ["param.json"]


def test_planted_dl_group_is_replaced():
    SIFEKeyGenerator(dict(DDH))

    def plant(record):
        record["group"] = {"p": "23", "q": "11", "r": "2"}
        record["g"] = "4"

    _rewrite(CryptoCONST.TYPE_SIFE, plant)
    kg = SIFEKeyGenerator(dict(DDH))
    assert int(kg.p).bit_length() == 128 and gp.is_prime(kg.p) and gp.powmod(kg.g, kg.q, kg.p) == 1


def test_dl_group_with_an_element_of_the_wrong_order_is_replaced():
    kg = SIFEKeyGenerator(dict(DDH))
    p = int(kg.p)
    _rewrite(CryptoCONST.TYPE_SIFE, lambda r: r.update(g=str(p - 1)))  # order 2
    assert SIFEKeyGenerator(dict(DDH)).g != p - 1


def test_planted_damgard_groups_are_replaced():
    cfg = {"sec_param": 128, "eta": 2, "modulus_length": 128, "bound": 10}
    SIFEDamgardKeyGenerator(dict(cfg))
    _rewrite(CryptoCONST.TYPE_SIFE_DAMGARD, lambda r: r.update(h="1"))
    assert SIFEDamgardKeyGenerator(dict(cfg)).h != 1
    mcfg = {"n": 2, "lst_nid": ["a", "b"], "eta": 1, "modulus_length": 128, "bound": 10}
    MCFEDamgardKeyGenerator(dict(mcfg))
    _rewrite(CryptoCONST.TYPE_MCFE_DAMGARD, lambda r: r.update(p="23", q="11", g="4"))
    assert int(MCFEDamgardKeyGenerator(dict(mcfg)).p).bit_length() == 128


def test_planted_dcr_moduli_are_replaced():
    cfg = {"sec_param": 64, "eta": 2, "bit_length": 64, "bound_x": 10, "bound_y": 10}
    SIFEPaillierKeyGenerator(dict(cfg))
    prime = int(gp.next_prime(2 ** 127))  # right size, but prime: no DCR

    def plant(record):
        record["n"], record["n_square"] = str(prime), str(prime * prime)

    _rewrite(CryptoCONST.TYPE_SIFE_PAILLIER, plant)
    assert int(SIFEPaillierKeyGenerator(dict(cfg)).n) != prime
    mcfg = {"n": 2, "lst_nid": ["a", "b"], "eta": 1, "bit_length": 64, "bound_x": 10, "bound_y": 10}
    MCFEPaillierKeyGenerator(dict(mcfg))
    _rewrite(CryptoCONST.TYPE_MCFE_PAILLIER, lambda r: r.update(n_mod="35"))
    assert int(MCFEPaillierKeyGenerator(dict(mcfg)).n_mod).bit_length() >= 127
