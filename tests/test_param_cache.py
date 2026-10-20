"""Where the parameter cache is (``<cache_dir>/authority``) and what it
accepts: a corrupted file is regenerated,
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
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.base import cache_dir
from pyfe4ai.core.exceptions import FEValidationError


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _param_file(scheme_type):
    return os.path.join(cache_dir(), "authority", scheme_type, "param.json")


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


def _prime_with_small_subgroup(bits, q_bits=32):
    """A bits-bit prime p = r q + 1 with a q_bits-bit prime q (r even)."""
    q = gp.next_prime(gp.mpz(1) << (q_bits - 1))
    r = (gp.mpz(1) << (bits - 1)) // q
    r += r % 2
    while True:
        p = r * q + 1
        if p.bit_length() == bits and gp.is_prime(p):
            return p, q, r
        r += 2


def test_planted_group_with_a_small_subgroup_is_replaced():
    """Regression: the cofactor r came from the cache and the size of q was not
    checked, so p = r q + 1 with a 32-bit q passed (discrete logs in the order-q
    subgroup then take seconds)."""
    SIFEKeyGenerator(dict(DDH))
    p, q, r = _prime_with_small_subgroup(128)
    h = gp.mpz(2)
    g = gp.powmod(h, r, p)

    def plant(record):
        record["group"] = {"p": str(p), "q": str(q), "r": str(r)}
        record["g"] = str(g)

    _rewrite(CryptoCONST.TYPE_SIFE, plant)
    kg = SIFEKeyGenerator(dict(DDH))
    assert int(kg.q).bit_length() == 127 and int(kg.r) == 2


def test_rfc_sizes_must_use_the_rfc_prime(monkeypatch):
    """At a size with an RFC 3526 prime, another safe prime (which could carry
    a trapdoor) is refused; checked at a small size registered for the test."""
    from pyfe4ai.primitives.groups import integer_groups
    from pyfe4ai.primitives.groups.integer_groups import _safe_prime, is_dl_group

    p1, q1 = _safe_prime(96)
    while True:
        p2, q2 = _safe_prime(96)
        if p2 != p1:
            break
    monkeypatch.setitem(integer_groups.RFC3526_PRIMES, 96, p1)
    assert is_dl_group(p1, q1, 4, 96)
    assert not is_dl_group(p2, q2, 4, 96)


def test_dcr_moduli_with_small_factors_and_paillier_bases_of_order_n_are_refused():
    from pyfe4ai.primitives.groups.integer_groups import is_dcr_modulus

    big = gp.next_prime(gp.mpz(1) << 126)
    assert not is_dcr_modulus(3 * big, 64)
    cfg = {"sec_param": 64, "eta": 2, "bit_length": 64, "bound_x": 10, "bound_y": 10}
    kg = SIFEPaillierKeyGenerator(dict(cfg))
    n = int(kg.n)
    _rewrite(CryptoCONST.TYPE_SIFE_PAILLIER, lambda r: r.update(g=str(1 + n)))
    assert int(SIFEPaillierKeyGenerator(dict(cfg)).g) % n != 1


@pytest.mark.parametrize("multi", [False, True])
def test_ring_caches_hold_a_seed_and_are_rederived(multi):
    """Regression: the public polynomial a and the widths came from the cache
    unchecked (a = 0 and sigma1 = 0 were accepted, also under standard)."""
    if multi:
        from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator as K
        cfg = {"sec_param": 64, "eta": 2, "n": 2, "lst_nid": ["a", "b"], "bound_x": 4, "bound_y": 4,
               "ring_n": 16}
        scheme_type = CryptoCONST.TYPE_MIFE_RING_LWE
    else:
        from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWEKeyGenerator as K
        cfg = {"sec_param": 64, "eta": 2, "bound_x": 4, "bound_y": 4, "ring_n": 16}
        scheme_type = CryptoCONST.TYPE_SIFE_RING_LWE
    kg = K(dict(cfg))
    with open(_param_file(scheme_type)) as f:
        record = json.load(f)
    assert "A" not in record and len(bytes.fromhex(record["A_seed"])) == 32
    for plant in ({"sigma1": 0.0}, {"q": str(int(record["q"]) + 32)}, {"p": "5"}):
        _rewrite(scheme_type, lambda r: r.update(plant))
        again = K(dict(cfg))
        assert (again.q, again.p, again.sigma1) == (kg.q, kg.p, kg.sigma1)


def test_adaptive_ring_cache_holds_a_seed():
    from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWEKeyGenerator as K

    cfg = {"eta": 2, "bound_x": 2, "bound_y": 2, "ring_n": 32}
    kg = K(dict(cfg))
    with open(_param_file(CryptoCONST.TYPE_SIFE_FULLYSEC_RING_LWE)) as f:
        record = json.load(f)
    assert "a" not in record and len(bytes.fromhex(record["a_seed"])) == 32
    again = K(dict(cfg))
    assert again.a == kg.a


# -- second check: everything a cache holds is re-derived or seeded -------


def test_abdp15_cache_is_rederived():
    """lwe_n, q, m and sigma_q were read from the file: lwe_n = 1 with
    sigma_q = 0, or m = 8, were accepted and leaked the plaintexts."""
    from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator as K

    cfg = {"sec_param": 64, "eta": 3, "lwe_n": 16, "bound_x": 8, "bound_y": 8}
    kg = K(dict(cfg))
    for plant in ({"lwe_n": 1, "sigma_q": 0.0}, {"m": 8}, {"q": "101"}):
        _rewrite(CryptoCONST.TYPE_SIFE_LWE, lambda r: r.update(plant))
        again = K(dict(cfg))
        assert (again.lwe_n, again.m, again.q, again.sigma_q) == (kg.lwe_n, kg.m, kg.q, kg.sigma_q)


def test_threshold_mife_lwe_flooding_width_is_rederived():
    from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator as K

    cfg = {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["a", "b"], "s": 3, "t": 2,
           "lst_sid": ["s0", "s1", "s2"], "lwe_n": 16, "bound_x": 4, "bound_y": 4}
    kg = K(dict(cfg))
    _rewrite(CryptoCONST.TYPE_TMIFE_LWE, lambda r: r.update(sigma_f="1"))
    assert K(dict(cfg)).sigma_f == kg.sigma_f > 1


def test_als16_cache_holds_a_seed_and_is_rederived():
    from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWEKeyGenerator as K

    cfg = {"sec_param": 64, "eta": 2, "lwe_n": 8, "bound_x": 4, "bound_y": 4}
    kg = K(dict(cfg))
    with open(_param_file(CryptoCONST.TYPE_SIFE_FULLYSEC_LWE)) as f:
        record = json.load(f)
    assert "A" not in record and len(bytes.fromhex(record["A_seed"])) == 32
    assert K(dict(cfg)).A == kg.A  # expanded from the stored seed
    _rewrite(CryptoCONST.TYPE_SIFE_FULLYSEC_LWE, lambda r: r.update(sigma_q=0.0))
    assert K(dict(cfg)).sigma_q == kg.sigma_q


def test_paillier_base_is_derived_from_a_seed():
    """A planted g = N^2 - 1 (order 2) made every ciphertext reveal x."""
    cfg = {"sec_param": 64, "eta": 2, "bit_length": 64, "bound_x": 10, "bound_y": 10}
    kg = SIFEPaillierKeyGenerator(dict(cfg))
    with open(_param_file(CryptoCONST.TYPE_SIFE_PAILLIER)) as f:
        record = json.load(f)
    assert "g" not in record and len(bytes.fromhex(record["g_seed"])) == 32
    n2 = int(kg.n) ** 2
    _rewrite(CryptoCONST.TYPE_SIFE_PAILLIER, lambda r: r.update(g=str(n2 - 1)))
    assert int(SIFEPaillierKeyGenerator(dict(cfg)).g) == int(kg.g) != n2 - 1


def test_damgard_h_is_derived_from_a_seed():
    """A planted h = g (known discrete log) was accepted."""
    cfg = {"sec_param": 128, "eta": 2, "modulus_length": 128, "bound": 10}
    kg = SIFEDamgardKeyGenerator(dict(cfg))
    with open(_param_file(CryptoCONST.TYPE_SIFE_DAMGARD)) as f:
        record = json.load(f)
    assert "h" not in record and len(bytes.fromhex(record["h_seed"])) == 32
    _rewrite(CryptoCONST.TYPE_SIFE_DAMGARD, lambda r: r.update(h=r["g"]))
    assert SIFEDamgardKeyGenerator(dict(cfg)).h == kg.h != kg.g


def test_standard_never_loads_or_stores_dcr_moduli(monkeypatch):
    """A cached DCR modulus cannot be checked for an unknown factorisation, so
    under standard it is generated afresh (toy keeps the cache)."""
    from pyfe4ai.core import config as configs
    from pyfe4ai.schemes.sife import paillier
    from pyfe4ai.primitives.groups.integer_groups import _safe_prime

    monkeypatch.setattr(paillier, "generate_group_primes", lambda bits: _safe_prime(64))
    monkeypatch.setitem(configs.STANDARD_MINIMA, "dcr", {"bit_length": 64, "stat_param": 128})
    std = {"eta": 2, "bound_x": 10, "bound_y": 10, "security_profile": "standard"}
    first = SIFEPaillierKeyGenerator(dict(std))
    assert not os.path.exists(_param_file(CryptoCONST.TYPE_SIFE_PAILLIER))
    assert SIFEPaillierKeyGenerator(dict(std)).n != first.n
    toy = {"sec_param": 64, "eta": 2, "bit_length": 64, "bound_x": 10, "bound_y": 10}
    assert SIFEPaillierKeyGenerator(dict(toy)).n == SIFEPaillierKeyGenerator(dict(toy)).n


# -- location of the cache --------------------------------------------------------


def test_cache_directory_precedence(tmp_path, monkeypatch):
    from pyfe4ai.core import base

    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "env"))
    assert base.cache_dir() == str(tmp_path / "env")
    assert base.cache_dir(tmp_path / "explicit") == str(tmp_path / "explicit")  # the config entry wins
    assert base.cache_dir("relative") == "relative"
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", "")
    assert base.cache_dir() == base.default_cache_dir()
    monkeypatch.delenv("PYFE4AI_CACHE_DIR")
    assert base.cache_dir() == base.default_cache_dir()
    for bad in ("", 5, b"bytes", ["dir"]):
        with pytest.raises(FEValidationError, match="cache_dir"):
            base.cache_dir(bad)


def test_default_cache_directory_per_platform(tmp_path, monkeypatch):
    from pyfe4ai.core import base

    home = str(tmp_path / "home")
    monkeypatch.setenv("HOME", home)
    with monkeypatch.context() as mp:
        mp.setattr(base.sys, "platform", "darwin")
        assert base.default_cache_dir() == os.path.join(home, "Library", "Caches", "pyfe4ai")
    with monkeypatch.context() as mp:
        mp.setattr(base.sys, "platform", "linux")
        mp.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert base.default_cache_dir() == os.path.join(str(tmp_path / "xdg"), "pyfe4ai")
        mp.setenv("XDG_CACHE_HOME", "relative/xdg")  # not absolute: ignored, as the XDG spec says
        assert base.default_cache_dir() == os.path.join(home, ".cache", "pyfe4ai")
        mp.delenv("XDG_CACHE_HOME")
        assert base.default_cache_dir() == os.path.join(home, ".cache", "pyfe4ai")
    with monkeypatch.context() as mp:
        mp.setattr(base.sys, "platform", "win32")
        mp.setattr(base.os, "name", "nt")
        mp.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        assert base.default_cache_dir() == os.path.join(str(tmp_path / "local"), "pyfe4ai", "Cache")


def test_records_go_to_the_cache_directory_not_the_working_directory(tmp_path, monkeypatch):
    from pyfe4ai.ml import FESchemeWrapper

    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "env"))
    SIFEKeyGenerator(dict(DDH))
    record = tmp_path / "env" / "authority" / CryptoCONST.TYPE_SIFE / "param.json"
    assert record.exists()
    SIFEKeyGenerator(dict(DDH, cache_dir=str(tmp_path / "explicit")))
    assert (tmp_path / "explicit" / "authority" / CryptoCONST.TYPE_SIFE / "param.json").exists()
    FESchemeWrapper("mcfe", "ddh", dict(DDH, n=2, cache_dir=str(tmp_path / "wrapper")))
    assert (tmp_path / "wrapper" / "authority" / CryptoCONST.TYPE_MCFE / "param.json").exists()
    assert not (tmp_path / "config").exists()  # the working directory
    # a record moved to another cache directory is loaded from there as it is
    moved = tmp_path / "moved" / "authority" / CryptoCONST.TYPE_SIFE
    moved.mkdir(parents=True)
    (moved / "param.json").write_bytes(record.read_bytes())
    kg = SIFEKeyGenerator(dict(DDH, cache_dir=str(tmp_path / "moved")))
    assert str(kg.p) == json.loads(record.read_text())["group"]["p"]
    assert (moved / "param.json").read_bytes() == record.read_bytes()
    with pytest.raises(FEValidationError, match="cache_dir"):
        SIFEKeyGenerator(dict(DDH, cache_dir=5))
