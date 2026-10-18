"""Replicated key sharing of the threshold LWE / Ring-LWE layers (``sharing = "replicated"``)."""

import random
from fractions import Fraction
from itertools import combinations
from math import comb

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator
from pyfe4ai.schemes.threshold_utils import pieces_held
from pyfe4ai.schemes.threshold_utils import replicated_assignment
from pyfe4ai.schemes.threshold_utils import replicated_sets
from pyfe4ai.schemes.threshold_utils import share_for_enrolled
from pyfe4ai.schemes.threshold_utils import share_replicated
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils import native_backend

Q = gp.next_prime(gp.mpz(2) ** 61)
ST = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 3), (5, 3), (5, 4)]


@pytest.mark.parametrize("s,t", ST)
def test_pieces_sum_to_the_secret_and_coalitions_miss_one(s, t):
    points = list(range(1, s + 1))
    secret = {"a": [5, Q - 1], "b": 7}
    pieces = share_replicated(secret, t, points, Q)
    assert set(pieces) == set(replicated_sets(points, t)) and len(pieces) == comb(s, t - 1)
    assert sum(gp.mpz(p["b"]) for p in pieces.values()) % Q == 7
    assert [sum(gp.mpz(p["a"][i]) for p in pieces.values()) % Q for i in (0, 1)] == [5, Q - 1]
    for x in points:
        assert len(pieces_held(pieces, x)) == comb(s - 1, t - 1)
    for size in range(1, s + 1):
        for group in combinations(points, size):
            held = {tuple(p["set"]) for x in group for p in pieces_held(pieces, x)}
            if size >= t:
                assert held == set(pieces)  # any t servers hold every piece
            elif size == t - 1:
                assert set(pieces) - held == {group}  # t - 1 miss exactly r_C


@pytest.mark.parametrize("s,t", ST)
def test_assignment_counts_every_piece_once_and_multipliers_are_0_or_1(s, t):
    points = list(range(1, s + 1))
    sets = replicated_sets(points, t)
    for size in range(t, s + 1):
        for enrolled in combinations(points, size):
            assigned = [replicated_assignment(k, list(enrolled), points, t) for k in enrolled]
            assert sorted(T for a in assigned for T in a) == sorted(sets)
            for k, a in zip(enrolled, assigned):
                assert all(k not in T for T in a)  # the server holds what it adds
            for coalition in combinations(points, t - 1):
                # M_k = [r_C is assigned to k]: one honest server of E carries nu
                carriers = [k for k, a in zip(enrolled, assigned) if coalition in a]
                assert len(carriers) == 1 and carriers[0] not in coalition


def test_shamir_multipliers_are_not_integers_for_t_below_s():
    """The gap the replicated sharing closes (``scratchpad/scripts/multipliers.py``)."""
    def lam(k, E):
        v = Fraction(1)
        for j in E:
            if j != k:
                v *= Fraction(j, j - k)
        return v

    def l0(k, C):
        v = Fraction(1)
        for j in C:
            v *= Fraction(k - j, 0 - j)
        return v

    counts = {}
    for s, t in [(3, 2), (3, 3), (5, 3)]:
        pts = range(1, s + 1)
        cases = [(C, E, k) for C in combinations(pts, t - 1) for size in range(t, s + 1)
                 for E in combinations(pts, size) for k in E if k not in C]
        counts[(s, t)] = (sum((lam(k, E) * l0(k, C)).denominator != 1 for C, E, k in cases), len(cases))
    assert counts == {(3, 2): (6, 18), (3, 3): (0, 3), (5, 3): (202, 330)}
    assert lam(1, (1, 3)) * l0(1, (2,)) == Fraction(3, 4)


def test_share_for_enrolled_sums_the_assigned_pieces():
    points, t = [1, 2, 3], 2
    pieces = share_replicated({"sk_y": [11, 12], "z": 13}, t, points, Q)
    dk = {"sharing": "replicated", "lst_sid": ["a", "b", "c"], "noise_key": "00",
          "pieces": pieces_held(pieces, 1)}
    share, multiplier, tag = share_for_enrolled(dk, "a", ["a", "c"], t, Q)
    assert multiplier == 1 and "pieces" not in share and share["noise_key"] == "00"
    want = [T for T in pieces if T != (1,)]  # E = {1, 3}: r_{2}, r_{3} go to server 1
    assert share["z"] == gp.digits(sum(gp.mpz(pieces[T]["z"]) for T in want) % Q)
    assert tag == '["replicated", [[2], [3]]]'
    share3, _, tag3 = share_for_enrolled(dict(dk, pieces=pieces_held(pieces, 3)), "c", ["a", "c"], t, Q)
    assert (gp.mpz(share["z"]) + gp.mpz(share3["z"])) % Q == 13 and tag3 == '["replicated", [[1]]]'
    # all three enrolled: server 3 adds nothing
    share3, _, tag3 = share_for_enrolled(dict(dk, pieces=pieces_held(pieces, 3)), "c", ["a", "b", "c"], t, Q)
    assert share3["z"] == "0" and share3["sk_y"] == ["0", "0"] and tag3 == '["replicated", []]'
    with pytest.raises(FEKeyError):
        share_for_enrolled(dict(dk, pieces=pieces_held(pieces, 3)), "a", ["a", "c"], t, Q)


# ------------------------------------------------------------------ schemes

def _enrolled_sets(sids, t):
    return [list(e) for size in range(t, len(sids) + 1) for e in combinations(sids, size)]


@pytest.mark.parametrize("s,t", [(3, 2), (4, 3), (3, 3), (2, 1)])
def test_tmife_lwe_replicated_decrypts_for_every_enrolled_set(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": s, "t": t, "lst_nid": ["c0", "c1"], "lst_sid": sids,
           "lwe_n": 16, "bound_x": 6, "bound_y": 6, "sharing": "replicated"}
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["sharing"] == "replicated"
    for sign in (1, -1):
        x = {nid: [sign * 6] for nid in cfg["lst_nid"]}
        y = {nid: [6] for nid in cfg["lst_nid"]}
        cts = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
               for nid in cfg["lst_nid"]}
        dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in sids}
        assert all(len(dk["pieces"]) == comb(s - 1, t - 1) and "sk_y" not in dk for dk in dks.values())
        groups = []
        for enrolled in _enrolled_sets(sids, t):
            groups.append({sid: ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
                cts, y, dks[sid], enrolled) for sid in enrolled})
            assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(groups[-1]) == sign * 72
        assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(groups) == sign * 72


def test_tmife_lwe_replicated_partials_are_derandomised_and_ndarrays_decrypt():
    sids = ["sid_0", "sid_1", "sid_2"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["c0", "c1"], "lst_sid": sids,
           "lwe_n": 16, "bound_x": 6, "bound_y": 6, "sharing": "replicated"}
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in cfg["lst_nid"]}
    x = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    cts = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
           for nid in cfg["lst_nid"]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    a = server.share_decrypt(cts, y, dk, ["sid_0", "sid_1"])
    assert server.share_decrypt(cts, y, dk, ["sid_0", "sid_1"]) == a
    # point 1 adds r_{2} and r_{3} for every enrolled set: one partial, one noise
    assert server.share_decrypt(cts, y, dk, ["sid_0", "sid_2"])["w"] == a["w"]
    assert server.share_decrypt(cts, y, dk, sids)["w"] == a["w"]
    # point 3 adds r_{1} for {1, 3} and nothing for {1, 2, 3}: two partials, fresh noise
    dk2 = kg.get_decryption_keys("sid_2", credentials={"fusion_weight": y})
    s2 = ThresholdMIFELWE({"id": "sid_2", "keys": {"pp": pp, "dk": dk2}})
    assert s2.share_decrypt(cts, y, dk2, ["sid_0", "sid_2"])["w"] != s2.share_decrypt(cts, y, dk2, sids)["w"]

    ary = {nid: [np.array([[1.0, -2.0], [3.0, 0.0]])] for nid in cfg["lst_nid"]}
    nd_cts = {nid: ThresholdMIFELWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
              .encrypt_lst_ndarray(ary[nid]) for nid in cfg["lst_nid"]}
    partials = {}
    for sid in ["sid_1", "sid_2"]:
        dk_s = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        partials[sid] = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk_s}}).compute_lst_ndarray_ct(
            nd_cts, y, dk_s, ["sid_1", "sid_2"])
    out = ThresholdMIFELWE({"id": "c", "precision": 0, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(partials)
    assert np.array_equal(out[0], np.array([[2.0, -4.0], [6.0, 0.0]]))


def test_sharing_must_match_the_public_parameters():
    sids = ["sid_0", "sid_1", "sid_2"]
    base = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["c0", "c1"], "lst_sid": sids,
            "lwe_n": 16, "bound_x": 6, "bound_y": 6}
    with pytest.raises(FEValidationError):
        ThresholdMIFELWEKeyGenerator(dict(base, sharing="additive"))
    kg = ThresholdMIFELWEKeyGenerator(dict(base, sharing="replicated"))
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [1] for nid in base["lst_nid"]}
    cts = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt([1])
           for nid in base["lst_nid"]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    shamir_pp = dict(pp, sharing="shamir")
    with pytest.raises(FESchemeError):
        ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": shamir_pp, "dk": dk}}).share_decrypt(cts, y, dk, sids[:2])
    # a server that lacks a piece it must add
    dk_bad = dict(dk, pieces=dk["pieces"][:1])
    with pytest.raises(FEKeyError):
        ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk_bad}}).share_decrypt(cts, y, dk_bad, sids[:2])


@pytest.mark.parametrize("s,t", [(3, 2), (4, 3)])
def test_tmcfe_ring_lwe_replicated_decrypts_for_every_enrolled_set(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": s, "t": t, "lst_nid": ["a", "b", "c"], "lst_sid": sids,
           "ring_n": 16, "bound_x": 4, "bound_y": 4, "sharing": "replicated"}
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    for sign in (1, -1):
        x = {nid: [[sign * 4] * 5] for nid in cfg["lst_nid"]}
        y = {nid: [4] for nid in cfg["lst_nid"]}
        label = "edge%d" % sign
        cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
               for nid in cfg["lst_nid"]}
        cred = {"fusion_weight": y, "label": label}
        dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
        assert all(len(dk["pieces"]) == comb(s - 1, t - 1) for dk in dks.values())
        for enrolled in _enrolled_sets(sids, t):
            group = {sid: ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
                cts, cred, dks[sid], enrolled) for sid in enrolled}
            assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == [sign * 48] * 5


def test_tmcfe_ring_lwe_replicated_ndarrays_decrypt():
    sids = ["sid_0", "sid_1", "sid_2"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["a", "b"], "lst_sid": sids,
           "ring_n": 16, "bound_x": 40, "bound_y": 4, "sharing": "replicated"}
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    y = {nid: [2] for nid in cfg["lst_nid"]}
    ary = {nid: [np.arange(40, dtype=float).reshape(5, 8) - 20] for nid in cfg["lst_nid"]}
    nd_cts = {nid: ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
              .encrypt_lst_ndarray(ary[nid], label="nd") for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "nd", "n_chunks": ThresholdMCFERingLWE.key_chunks(nd_cts["a"])}
    partials = {}
    for sid in ["sid_0", "sid_2"]:
        dk = kg.get_decryption_keys(sid, credentials=cred)
        partials[sid] = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).compute_lst_ndarray_ct(
            nd_cts, cred, dk, ["sid_0", "sid_2"])
    out = ThresholdMCFERingLWE({"id": "c", "precision": 0, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(partials)
    assert np.array_equal(out[0], 4 * (np.arange(40, dtype=float).reshape(5, 8) - 20))


@pytest.mark.parametrize("s,t", [(3, 2), (4, 3)])
def test_tmcfe_lwe_replicated_decrypts_for_every_enrolled_set(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": s, "t": t, "lst_nid": ["c0", "c1", "c2"], "lst_sid": sids,
           "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2, "label_modulus": 8,
           "sharing": "replicated"}
    kg = ThresholdMCFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    x = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    y = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    cts = {nid: ThresholdMCFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], "lbl")
           for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "lbl"}
    dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
    assert all(len(dk["pieces"]) == comb(s - 1, t - 1) and "sk_y" not in dk for dk in dks.values())
    want = sum(x[n][0] * y[n][0] for n in x)
    groups = []
    for enrolled in _enrolled_sets(sids, t):
        groups.append({sid: ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
            cts, cred, dks[sid], enrolled) for sid in enrolled})
        assert ThresholdMCFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(groups[-1]) == want
    assert ThresholdMCFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(groups) == want

    ary = {nid: [np.array([1.0, -2.0, 3.0])] for nid in cfg["lst_nid"]}
    ones = {nid: [1] for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": ones, "label": "nd"}
    nd_cts = {nid: ThresholdMCFELWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
              .encrypt_lst_ndarray(ary[nid], label="nd") for nid in cfg["lst_nid"]}
    partials = {}
    for sid in sids[:t]:
        dk = kg.get_decryption_keys(sid, credentials=cred)
        partials[sid] = ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).compute_lst_ndarray_ct(
            nd_cts, cred, dk, sids[:t])
    out = ThresholdMCFELWE({"id": "c", "precision": 0, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(partials)
    assert np.array_equal(out[0], np.array([3.0, -6.0, 9.0]))


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
def test_tmcfe_ring_lwe_replicated_partials_agree_across_backends():
    sids = ["sid_0", "sid_1", "sid_2"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["a", "b"], "lst_sid": sids,
           "ring_n": 16, "bound_x": 4, "bound_y": 4, "sharing": "replicated"}
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    with native_backend.backend("native"):
        assert native_backend.ring_context(gp.mpz(pp["q"]), pp["ring_n"]) is not None
    y = {nid: [3] for nid in cfg["lst_nid"]}
    cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(
        [[1, -2, 3]], "x") for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "x"}
    for enrolled in _enrolled_sets(sids, 2):
        for sid in enrolled:
            dk = kg.get_decryption_keys(sid, credentials=cred)
            server = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}})
            with native_backend.backend("python"):
                a = server.share_decrypt(cts, cred, dk, enrolled)
            with native_backend.backend("native"):
                b = server.share_decrypt(cts, cred, dk, enrolled)
            assert [str(v) for v in a["w"]] == [str(v) for v in b["w"]]


# ------------------------------------------------- default and refusal

_CONFIGS = {
    "mife": (ThresholdMIFELWEKeyGenerator, {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["c0", "c1"],
                                            "lwe_n": 16, "bound_x": 6, "bound_y": 6}),
    "ring": (ThresholdMCFERingLWEKeyGenerator, {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["a", "b"],
                                                "ring_n": 16, "bound_x": 4, "bound_y": 4}),
    "lwe": (ThresholdMCFELWEKeyGenerator, {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["c0", "c1"],
                                           "lwe_n": 16, "bound_x": 6, "bound_y": 6}),
}


def _make(kind, s, t, **extra):
    cls, base = _CONFIGS[kind]
    return cls(dict(base, s=s, t=t, lst_sid=["sid_%d" % i for i in range(s)], **extra))


@pytest.mark.parametrize("kind", sorted(_CONFIGS))
def test_default_sharing_is_replicated_below_s_and_shamir_at_s(kind):
    import warnings

    from pyfe4ai.utils.exceptions import FESecurityWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", FESecurityWarning)
        for s, t, want in [(3, 2, "replicated"), (3, 1, "replicated"), (2, 1, "replicated"),
                           (3, 3, "shamir"), (2, 2, "shamir"), (1, 1, "shamir")]:
            kg = _make(kind, s, t)
            kg.setup()
            assert kg.sharing == want and kg.get_public_parameters()["sharing"] == want, (s, t)
        _make(kind, 3, 3, sharing="shamir")  # explicit Shamir at t = s: no warning
        _make(kind, 3, 3, sharing="replicated")


@pytest.mark.parametrize("kind", sorted(_CONFIGS))
def test_explicit_shamir_below_s_warns_under_toy_and_is_refused_under_standard(kind):
    from pyfe4ai.utils.exceptions import FESecurityWarning

    with pytest.warns(FESecurityWarning, match="t = 2 < s = 3"):
        kg = _make(kind, 3, 2, sharing="shamir")
    assert kg.sharing == "shamir"
    # refused before any parameter is derived (standard sizes would take long)
    with pytest.raises(FEValidationError, match="Shamir sharing with t = 2 < s = 3"):
        _make(kind, 3, 2, sharing="shamir", security_profile="standard", sec_param=128)


def test_the_sharings_have_their_own_parameter_records():
    """A record cached for Shamir with t < s (versions <= 0.3.0) is not taken
    for the default; Shamir records at t = s are those of before."""
    import json
    import os

    from pyfe4ai.utils.exceptions import FESecurityWarning

    def record(kg):
        with open(os.path.join(kg.config_folder, kg._scheme_type, "param.json"), encoding="utf-8") as f:
            return json.load(f)

    with pytest.warns(FESecurityWarning):
        old_kg = _make("mife", 3, 2, sharing="shamir")
    old = record(old_kg)
    assert old["construction"] == "acfgu18-abdp15-shamir-smudging-seeded-a/v3"
    new_kg = _make("mife", 3, 2)
    new = record(new_kg)
    assert new["construction"] == "acfgu18-abdp15-replicated-smudging-seeded-a/v1"
    assert new["A_seed"] != old["A_seed"] and new_kg.q == old_kg.q  # regenerated, same parameters
    at_s = _make("mife", 3, 3)
    assert record(at_s)["construction"] == "acfgu18-abdp15-shamir-smudging-seeded-a/v3"

    with pytest.warns(FESecurityWarning):
        old_kg = _make("ring", 3, 2, sharing="shamir")
    old = record(old_kg)
    assert old["threshold_construction"] == "bmms21-ibmcfe-shamir-smudging/v1"
    new_kg = _make("ring", 3, 2)
    new = record(new_kg)
    assert new["threshold_construction"] == "bmms21-ibmcfe-replicated-smudging/v1"
    assert new["A_seed"] != old["A_seed"] and new_kg.q == old_kg.q


# --------------------------------------- review 2026-10: F4, F5, F7

@pytest.mark.parametrize("kind", sorted(_CONFIGS))
@pytest.mark.parametrize("t", [True, False, 2.0, None])
def test_a_threshold_must_be_an_integer(kind, t):
    with pytest.raises(FEValidationError):
        _make(kind, 3, t)


def test_validate_threshold_refuses_bool():
    from pyfe4ai.schemes.threshold_utils import validate_threshold

    with pytest.raises(FEValidationError):
        validate_threshold(True, ["a", "b"])
    validate_threshold(1, ["a", "b"])


def _v030(kg, sid, credentials):
    """pp and a key share in the format of v0.3.0 (no ``sharing`` entry)."""
    pp = {k: v for k, v in kg.get_public_parameters().items() if k != "sharing"}
    dk = {k: v for k, v in kg.get_decryption_keys(sid, credentials=credentials).items() if k != "sharing"}
    return pp, dk


def _served(kind, s, t, sharing=None, v030=False):
    """(scheme class, pp, key share of sid_0, ciphertexts, credentials) for one round."""
    import warnings

    cls = {"mife": ThresholdMIFELWE, "ring": ThresholdMCFERingLWE, "lwe": ThresholdMCFELWE}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kg = _make(kind, s, t, **({"sharing": sharing} if sharing else {}))
    kg.setup()
    y = {nid: [1] for nid in kg.lst_nid}
    cred = y if kind == "mife" else {"fusion_weight": y, "label": "L"}
    keycred = {"fusion_weight": y} if kind == "mife" else cred
    if v030:
        pp, dk = _v030(kg, "sid_0", keycred)
        pp.pop("threshold_construction"), pp.pop("security_profile")
    else:
        pp, dk = kg.get_public_parameters(), kg.get_decryption_keys("sid_0", credentials=keycred)
    x = [1] if kind != "ring" else [[1]]
    if kind == "mife":
        cts = {nid: cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x) for nid in y}
    else:
        cts = {nid: cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x, "L") for nid in y}
    return cls, pp, dk, cts, cred


@pytest.mark.parametrize("kind", sorted(_CONFIGS))
@pytest.mark.parametrize("v030", [True, False], ids=["v030-pp", "pp"])
def test_shamir_key_shares_with_t_below_s_are_refused_unless_legacy_warn(kind, v030):
    from pyfe4ai.utils.exceptions import FESecurityWarning

    cls, pp, dk, cts, cred = _served(kind, 3, 2, sharing="shamir", v030=v030)
    with pytest.raises(FESchemeError, match="re-share the threshold keys"):
        cls({"id": "sid_0", "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, cred, dk, ["sid_0", "sid_2"])
    with pytest.raises(FESchemeError):
        cls({"id": "sid_0", "legacy_shamir": "refuse", "keys": {"pp": pp, "dk": dk}}).share_decrypt(
            cts, cred, dk, ["sid_0", "sid_2"])
    server = cls({"id": "sid_0", "legacy_shamir": "warn", "keys": {"pp": pp, "dk": dk}})
    with pytest.warns(FESecurityWarning, match="t = 2 < s = 3"):
        server.share_decrypt(cts, cred, dk, ["sid_0", "sid_2"])
    # public parameters that record the standard profile: refused even so
    standard = dict(pp, security_profile="standard")
    with pytest.raises(FESchemeError, match="not under the standard profile"):
        cls({"id": "sid_0", "legacy_shamir": "warn", "keys": {"pp": standard, "dk": dk}}).share_decrypt(
            cts, cred, dk, ["sid_0", "sid_2"])
    with pytest.raises(FEValidationError):
        cls({"id": "sid_0", "legacy_shamir": "allow", "keys": {"pp": pp, "dk": dk}})


@pytest.mark.parametrize("kind", sorted(_CONFIGS))
@pytest.mark.parametrize("v030", [True, False], ids=["v030-pp", "pp"])
def test_shamir_key_shares_at_t_equal_s_are_served_silently(kind, v030):
    import warnings

    from pyfe4ai.utils.exceptions import FESecurityWarning

    cls, pp, dk, cts, cred = _served(kind, 3, 3, v030=v030)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FESecurityWarning)
        cls({"id": "sid_0", "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, cred, dk, ["sid_0", "sid_1", "sid_2"])


@pytest.mark.parametrize("kind", sorted(_CONFIGS))
def test_public_parameters_record_the_construction_and_profile(kind):
    want = {
        "mife": ("acfgu18-abdp15-shamir-smudging-seeded-a/v3", "acfgu18-abdp15-replicated-smudging-seeded-a/v1"),
        "ring": ("bmms21-ibmcfe-shamir-smudging/v1", "bmms21-ibmcfe-replicated-smudging/v1"),
        "lwe": ("abmpsw20-shamir-flooding/v1", "abmpsw20-replicated-flooding/v1"),
    }[kind]
    for t, construction in ((3, want[0]), (2, want[1])):
        cls, pp, dk, cts, cred = _served(kind, 3, t)
        assert pp["threshold_construction"] == construction and pp["security_profile"] == "toy"
        enrolled = ["sid_0", "sid_1", "sid_2"][:t]
        server = cls({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
        server.share_decrypt(cts, cred, dk, enrolled)
        # a construction this version does not know (a later one, say), or
        # one of the other sharing: refused, not guessed
        for other in ("x-later-construction/v9", want[1 - (t == 2)]):
            bad = dict(pp, threshold_construction=other)
            with pytest.raises(FESchemeError, match="unknown threshold construction"):
                cls({"id": "sid_0", "keys": {"pp": bad, "dk": dk}}).share_decrypt(cts, cred, dk, enrolled)
        # no construction id: only Shamir public parameters of v0.3.0 lack it
        bare = {k: v for k, v in pp.items() if k != "threshold_construction"}
        if t == 2:
            with pytest.raises(FESchemeError, match="must record threshold_construction"):
                cls({"id": "sid_0", "keys": {"pp": bare, "dk": dk}}).share_decrypt(cts, cred, dk, enrolled)
        else:
            cls({"id": "sid_0", "keys": {"pp": bare, "dk": dk}}).share_decrypt(cts, cred, dk, enrolled)


def test_summed_shares_are_cached_per_key_share_and_assignment():
    from collections import OrderedDict

    from pyfe4ai.schemes import threshold_utils

    points, t = [1, 2, 3], 2
    pieces = share_replicated({"sk_y": [11, 12], "z": 13, "z_chunks": [[1, 2], [3, 4]]}, t, points, Q)
    dk = {"sharing": "replicated", "lst_sid": ["a", "b", "c"], "pieces": pieces_held(pieces, 1)}
    cache = OrderedDict()
    first = share_for_enrolled(dk, "a", ["a", "b"], t, Q, cache=cache)
    assert share_for_enrolled(dk, "a", ["a", "b"], t, Q, cache=cache) is first
    # same assigned pieces for {1, 3}: the same entry
    assert share_for_enrolled(dk, "a", ["a", "c"], t, Q, cache=cache) is first and len(cache) == 1
    # another key share object (even with equal content) is summed again
    again = share_for_enrolled(dict(dk), "a", ["a", "b"], t, Q, cache=cache)
    assert again is not first and again == first
    # only the requested fields
    part = share_for_enrolled(dk, "a", ["a", "b"], t, Q, fields=("sk_y", "z"), cache=cache)
    assert "z_chunks" not in part[0] and part[0]["z"] == first[0]["z"] and part[0]["sk_y"] == first[0]["sk_y"]
    for i in range(threshold_utils.SHARE_CACHE_SIZE + 5):
        share_for_enrolled(dict(dk), "a", ["a", "b"], t, Q, cache=cache)
    assert len(cache) == threshold_utils.SHARE_CACHE_SIZE
    assert share_for_enrolled(dk, "a", ["a", "b"], t, Q) == first  # no cache: same values
