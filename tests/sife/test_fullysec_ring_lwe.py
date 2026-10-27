"""Adaptively secure Ring-LWE IPFE (BMMS21 Sec. 5) with re-derived parameters."""

import math
import os
import random

import gmpy2 as gp
import pytest

from pyfe4ai.core import config as configs
from pyfe4ai.schemes.sife.fullysec_ring_lwe import (
    SIFEFullySecRingLWE,
    SIFEFullySecRingLWEKeyGenerator,
    encode_message,
    independent_at_every_slot,
)
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.ring_lwe_utils import (
    _bmms21_adaptive_slack,
    bmms21_adaptive_log2_q,
    bmms21_adaptive_parameters,
    check_ring_lwe,
    ring_lwe_bound,
)
from pyfe4ai.core.base import cache_dir

CFG = {"eta": 3, "bound_x": 4, "bound_y": 4}


@pytest.fixture(autouse=True)
def _param_cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


@pytest.fixture(scope="module")
def scheme(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("cache"))
        mp.setenv("PYFE4AI_CACHE_DIR", os.path.join(os.getcwd(), "config"))
        kg = SIFEFullySecRingLWEKeyGenerator(dict(CFG))
        kg.setup()
    pp = kg.get_public_parameters()
    return kg, SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}}), pp


# -- parameters ---------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4])
@pytest.mark.parametrize("args", [(128, 10, 50, 50, 8192), (128, 3, 4, 4, 32)])
def test_three_or_four_key_polynomials_admit_no_modulus(m, args):
    # Ring-LHL forces sigma' >= q^(2/m), so correctness would need q > c q^(4/m)
    assert all(_bmms21_adaptive_slack(*args, m, float(lq)) < 0 for lq in range(20, 4001, 20))
    assert bmms21_adaptive_log2_q(*args, m) is None
    assert bmms21_adaptive_log2_q(*args, 5) is not None


@pytest.mark.parametrize("kappa, l, X, Y, n, m", [
    (128, 10, 50, 50, 8192, 11),
    (128, 4, 4, 4, 4096, 20),
    (128, 3, 4, 4, 32, 8),
])
def test_parameters_meet_the_constraints_of_appendix_c1(kappa, l, X, Y, n, m):
    P = bmms21_adaptive_parameters(kappa, l, X, Y, n, m)
    q, p = int(P["q"]), int(P["p"])
    assert P["m"] == m and q % (2 * n) == 1 and gp.is_prime(q)
    log2_lev = 2 * l * math.log2(2 * X + 1)  # complexity leveraging over (2X + 1)^(2l)
    # Theorem 2 (k = 2): the width sigma' fixes delta and eps through
    # sigma' = sqrt(n ln(2mn(1 + 1/delta)) / pi) q^(2/m + eps/2); read eps off with
    # the delta the bound below needs, then check the whole budget
    log2_delta = -(kappa + 3 + log2_lev)
    ln_term = math.log(2 * m * n) + math.log1p(2.0 ** -log2_delta)
    log2_root = 0.5 * math.log2(n * ln_term / math.pi)
    eps = 2 * (math.log2(P["sigma_lhl"]) - log2_root) / math.log2(q) - 4 / m
    bad = (2 * m - math.log2((q ** m - 1) * (q ** m - q)) / math.log2(q))  # log_q of the ratio
    log2_bad = n - eps * n * math.log2(q) + n * bad * math.log2(q)
    budget = math.log2(2 * 2.0 ** log2_delta + 2.0 ** log2_bad) + log2_lev
    assert budget <= -kappa
    # G3 -> G4 (Lemma 1 with alpha = x1 - x0, |alpha_i| <= 2X): Gamma >= sigma' / sqrt(2)
    assert P["sigma1"] >= math.sqrt(2) * 2 * X * math.sqrt(l) * P["sigma_lhl"] * (1 - 1e-12)
    # Theorem 1 (mhe-RLWE from RLWE with width sigma), C = sqrt(kappa) sigma1
    C = math.sqrt(kappa) * P["sigma1"]
    eta = math.sqrt((math.log(2 * (n + n * l)) + kappa * math.log(2)) / math.pi)
    ratio = P["sigma"] * n * C * math.sqrt(l + 2) / P["sigma2"]
    assert P["sigma"] * math.sqrt(1 - ratio ** 2) > eta
    assert P["sigma3"] == pytest.approx(math.sqrt(m) * P["sigma2"])
    # correctness (Appendix C.1) with the signed plaintext modulus p = 2 l X Y + 2
    assert p == 2 * l * X * Y + 2
    noise = l * Y * (math.sqrt(kappa) * P["sigma3"] + m * n * kappa * P["sigma1"] * P["sigma2"])
    assert noise < q // (2 * p)


def test_explicit_three_key_polynomials_are_refused():
    with pytest.raises(FEValidationError, match="m = 3"):
        SIFEFullySecRingLWEKeyGenerator({**CFG, "m": 3})


def test_auto_m_is_the_smallest_that_admits_a_modulus(scheme):
    kg, _, _ = scheme
    assert kg.m >= 5
    for m in range(5, kg.m):
        assert bmms21_adaptive_log2_q(kg.sec_param, CFG["eta"], 4, 4, kg.ring_n, m) is None


def test_message_encoding_is_linear():
    # the proof's last step needs floor(q/K) (x0 + u alpha), i.e. a linear encoding
    q, p = 2 ** 61 - 1, 98
    for a, b in ((3, -4), (-4, -4), (4, 1)):
        assert (encode_message(a, q, p) + encode_message(b, q, p)) % q == encode_message(a + b, q, p)
    assert encode_message(1, q, p) == q // p


def test_oversized_bounds_are_refused_cleanly():
    with pytest.raises(FEValidationError):
        bmms21_adaptive_parameters(128, 200, 2 ** 20, 2 ** 20, 32, 5)


# -- scheme -------------------------------------------------------------------


def test_round_trip_with_extreme_values(scheme):
    kg, enc, pp = scheme
    dec = SIFEFullySecRingLWE({"keys": {"pp": pp}})
    for x, y in (([4, -4, 3], [-4, 4, 2]), ([4, 4, 4], [4, 4, 4]), ([-4, -4, -4], [4, 4, 4])):
        dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
        assert dec.decrypt(enc.encrypt(x), dk, y) == sum(a * b for a, b in zip(x, y))


def test_random_inner_products(scheme):
    kg, enc, pp = scheme
    dec = SIFEFullySecRingLWE({"keys": {"pp": pp}})
    rng = random.Random(1)
    for _ in range(5):
        x = [rng.randint(-4, 4) for _ in range(3)]
        y = [rng.randint(-4, 4) for _ in range(3)]
        dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
        assert dec.decrypt(enc.encrypt(x), dk, y) == sum(a * b for a, b in zip(x, y))


def test_weights_are_bound_to_the_key_and_inputs_validated(scheme):
    kg, enc, pp = scheme
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1, 2, 3]})
    ct = enc.encrypt([1, 1, 1])
    with pytest.raises(FESchemeError):
        SIFEFullySecRingLWE({"keys": {"pp": pp}}).decrypt(ct, dk, [1, 2, 4])
    with pytest.raises(FEValidationError):
        enc.encrypt([5, 0, 0])
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("sid", credentials={"fusion_weight": [5, 0, 0]})


def test_key_and_ciphertext_have_m_polynomials(scheme):
    kg, enc, pp = scheme
    ct = enc.encrypt([1, 2, 3])
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [1, 0, 0]})
    assert len(ct["ct0"]) == len(dk["sk_y"]) == len(pp["a"]) == kg.m >= 5
    assert len(ct["ct"]) == CFG["eta"]


def test_slot_independence_detects_a_multiple_of_a():
    # m = 3 elements over 4 NTT slots mod 97; "slot t" is column t
    q = 97
    a = [[3, 5, 7, 11], [2, 4, 6, 8], [1, 1, 1, 1]]
    multiple = [[(9 * v) % q for v in row] for row in a]  # c = 9 a: dependent everywhere
    assert not independent_at_every_slot(a, multiple, q)
    partly = [row[:] for row in multiple]
    partly[2][1] = (partly[2][1] + 1) % q  # independent in slot 1 only
    assert not independent_at_every_slot(a, partly, q)
    other = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]  # no slot proportional to a
    assert independent_at_every_slot(a, other, q)


# -- profile ------------------------------------------------------------------


def test_standard_profile_presets_ring_parameters():
    out = configs.resolve({"security_profile": "standard"}, ("sife", "fullysec_ring_lwe"))
    assert out["ring_n"] == "auto" and out["sec_param"] == 128


def test_standard_picks_an_estimated_ring_and_at_least_five_polynomials():
    kg = SIFEFullySecRingLWEKeyGenerator({"eta": 4, "bound_x": 4, "bound_y": 4,
                                          "security_profile": "standard"})
    assert kg.m >= 5
    check_ring_lwe(kg.ring_n, kg.q, kg.sigma)


def test_cached_parameters_are_reloaded_and_checked():
    kg = SIFEFullySecRingLWEKeyGenerator(dict(CFG))
    again = SIFEFullySecRingLWEKeyGenerator(dict(CFG))
    assert (again.q, again.m, again.a) == (kg.q, kg.m, kg.a)
    import json, os
    path = os.path.join(cache_dir(), "authority", "SIFE_FULLYSEC_RING_LWE", "param.json")
    record = json.load(open(path))
    record["q"] = str(int(record["q"]) + 2 * kg.ring_n)  # a planted modulus
    json.dump(record, open(path, "w"))
    regenerated = SIFEFullySecRingLWEKeyGenerator(dict(CFG))
    assert regenerated.q == kg.q and regenerated.a != kg.a


def test_standard_round_trip():
    kg = SIFEFullySecRingLWEKeyGenerator({"eta": 4, "bound_x": 4, "bound_y": 4,
                                          "security_profile": "standard"})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": [4, -4, 4, -4]})
    assert SIFEFullySecRingLWE({"keys": {"pp": pp}}).decrypt(enc.encrypt([4, 4, -4, -4]), dk,
                                                             [4, -4, 4, -4]) == 0


def test_standard_with_an_explicit_ring_picks_an_m_that_meets_the_bound():
    """m = "auto" took m = 5 for an explicit ring_n under standard, whose q is
    above the bound, and refused although a larger m fits."""
    kg = SIFEFullySecRingLWEKeyGenerator({"eta": 4, "bound_x": 4, "bound_y": 4, "ring_n": 4096,
                                          "security_profile": "standard"})
    assert kg.ring_n == 4096 and kg.m > 5
    check_ring_lwe(kg.ring_n, kg.q, kg.sigma)
    assert bmms21_adaptive_log2_q(kg.sec_param, 4, 4, 4, 4096, kg.m - 1) > ring_lwe_bound(4096, kg.sigma) - 1
