"""Seeded replicated key sharing of the threshold LWE / Ring-LWE layers (``sharing = "replicated-seeded"``)."""

import json
import random
from itertools import combinations
from math import comb

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import lwe_threshold as tmcfe_lwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe_ring
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife import lwe_threshold as tmife_lwe
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator
from pyfe4ai.schemes.threshold_utils import SEED_BYTES
from pyfe4ai.schemes.threshold_utils import expand_seeded_piece
from pyfe4ai.schemes.threshold_utils import piece_layout
from pyfe4ai.schemes.threshold_utils import replicated_sets
from pyfe4ai.schemes.threshold_utils import seeded_pieces_held
from pyfe4ai.schemes.threshold_utils import share_for_enrolled
from pyfe4ai.schemes.threshold_utils import share_replicated_seeded
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError
from pyfe4ai.utils import native_backend

Q = gp.next_prime(gp.mpz(2) ** 61)
ST = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 3), (5, 3), (5, 4)]
SECRET = {"a": [5, Q - 1, 0], "b": 7, "c": [[1, 2], [3]]}


def _digits(material):
    if isinstance(material, dict):
        return {k: _digits(v) for k, v in material.items()}
    if isinstance(material, list):
        return [_digits(v) for v in material]
    return gp.digits(gp.mpz(material))


def _pieces(sharing, q):
    """Every piece r_T of a seeded sharing, expanded, as decimal strings."""
    T0, correction = sharing["correction"]
    out = {T0: correction}
    for T, seed in sharing["seeds"].items():
        out[T] = _digits(expand_seeded_piece(seed, T, sharing["layout"], q))
    return out


def _full_pp(pp, module):
    """*pp* as the replicated sharing publishes it (same parameters)."""
    assert pp["threshold_construction"] == module._CONSTRUCTION_REPLICATED_SEEDED
    return dict(pp, sharing="replicated", threshold_construction=module._CONSTRUCTION_REPLICATED)


def _expanded(dk, q):
    """The replicated key share (``sharing = "replicated"``) holding the expanded pieces of *dk*."""
    pieces = [dict(p) for p in dk["pieces"]] + [
        {"set": e["set"], "share": _digits(expand_seeded_piece(e["seed"], e["set"], dk["layout"], q))}
        for e in dk["seeds"]
    ]
    out = {k: v for k, v in dk.items() if k not in ("seeds", "layout")}
    out.update(sharing="replicated", pieces=pieces)
    return out


@pytest.mark.parametrize("s,t", ST)
def test_seeded_pieces_sum_to_the_secret_and_servers_hold_at_most_one_piece(s, t):
    points = list(range(1, s + 1))
    sharing = share_replicated_seeded(SECRET, t, points, Q)
    sets = replicated_sets(points, t)
    T0 = sharing["correction"][0]
    assert T0 == sets[0] == tuple(range(1, t)) and set(sharing["seeds"]) == set(sets[1:])
    assert all(len(bytes.fromhex(seed)) == SEED_BYTES for seed in sharing["seeds"].values())
    pieces = _pieces(sharing, Q)
    assert sum(gp.mpz(p["b"]) for p in pieces.values()) % Q == 7
    assert [sum(gp.mpz(p["a"][i]) for p in pieces.values()) % Q for i in range(3)] == [5, Q - 1, 0]
    assert sum(gp.mpz(p["c"][1][0]) for p in pieces.values()) % Q == 3
    for x in points:
        held = seeded_pieces_held(sharing, x)
        assert len(held["pieces"]) == (0 if x < t else 1)  # servers 1..t-1 hold seeds only
        assert len(held["pieces"]) + len(held["seeds"]) == comb(s - 1, t - 1)
        assert all(x not in e["set"] for e in held["pieces"] + held["seeds"])
    for size in range(1, s + 1):
        for group in combinations(points, size):
            held = set()
            for x in group:
                h = seeded_pieces_held(sharing, x)
                held |= {tuple(e["set"]) for e in h["pieces"] + h["seeds"]}
            if size >= t:
                assert held == set(sets)  # any t servers produce every piece
            elif size == t - 1:
                assert set(sets) - held == {group}  # t - 1 miss exactly r_C (its seed, or r_T0)


def test_expansion_is_deterministic_and_domain_separated():
    layout = piece_layout(SECRET)
    assert layout == {"a": 3, "b": None, "c": [2, 1]}
    seed = "ab" * SEED_BYTES
    one = expand_seeded_piece(seed, (2,), layout, Q)
    assert one == expand_seeded_piece(seed, (2,), json.loads(json.dumps(layout)), Q)
    assert all(0 <= v < Q for v in one["a"]) and 0 <= one["b"] < Q
    assert one != expand_seeded_piece(seed, (3,), layout, Q)  # the set is bound
    assert one["a"][:2] != one["c"][0]  # and the path
    other_q = expand_seeded_piece(seed, (2,), layout, gp.next_prime(Q))
    assert [v % Q for v in other_q["a"]] != one["a"]  # and the modulus
    assert expand_seeded_piece(seed, (2,), layout, Q, fields=("b",)) == {"b": one["b"]}
    for bad in ["ab" * (SEED_BYTES - 1), "zz" * SEED_BYTES, None]:
        with pytest.raises(FEKeyError):
            expand_seeded_piece(bad, (2,), layout, Q)
    with pytest.raises(FEKeyError):
        expand_seeded_piece(seed, (2,), {"a": -1}, Q)


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
def test_native_expansion_gives_the_python_values():
    from pyfe4ai.utils.ring_lwe_utils import next_ntt_prime

    q = next_ntt_prime(gp.mpz(2) ** 200, 16)
    with native_backend.backend("native"):
        ctx = native_backend.ring_context(q, 16)
    assert ctx is not None
    layout = {"sk_y": {"a": 16, "b": 16}, "z": 16, "z_chunks": [16, 16], "s": None, "e": 0}
    seed = "5c" * SEED_BYTES
    assert expand_seeded_piece(seed, (1, 3), layout, q, ctx=ctx) == expand_seeded_piece(seed, (1, 3), layout, q)
    # sums (and the dealer's differences) of explicit and seeded pieces, natively or in Python
    from pyfe4ai.schemes.threshold_utils import _combine_pieces

    material = {"sk_y": {"a": list(range(-8, 8)), "b": [str(q - 1)] * 16}, "z": [q + 5] * 16,
                "z_chunks": [[1] * 16, [0] * 16], "s": -3, "e": []}
    seeded = [("5c" * SEED_BYTES, (1, 3)), ("a7" * SEED_BYTES, (2, 3))]
    for subtract in (False, True):
        want = _combine_pieces(layout, [material, material], seeded, q, subtract=subtract)
        assert _combine_pieces(layout, [material, material], seeded, q, ctx, subtract=subtract) == want
    assert want["z"][0] == gp.digits((10 - sum(expand_seeded_piece(sd, T, layout, q)["z"][0] for sd, T in seeded)) % q)


@pytest.mark.parametrize("s,t", ST)
def test_share_for_enrolled_matches_the_replicated_share_of_the_expanded_pieces(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    points = list(range(1, s + 1))
    sharing = share_replicated_seeded(SECRET, t, points, Q)
    for size in range(t, s + 1):
        for enrolled in combinations(sids, size):
            total = gp.mpz(0)
            for k, sid in enumerate(sids, 1):
                if sid not in enrolled:
                    continue
                dk = {"sharing": "replicated-seeded", "lst_sid": sids, "noise_key": "00",
                      **seeded_pieces_held(sharing, k)}
                share, multiplier, tag = share_for_enrolled(dk, sid, list(enrolled), t, Q)
                want, want_multiplier, want_tag = share_for_enrolled(_expanded(dk, Q), sid, list(enrolled), t, Q)
                assert (multiplier, tag) == (want_multiplier, want_tag) == (1, tag)
                assert dict(share, sharing="replicated") == want
                assert not {"pieces", "seeds", "layout"} & set(share)
                total += gp.mpz(share["b"])
            assert total % Q == 7


def test_seeded_key_shares_refuse_missing_or_malformed_seeds():
    sids = ["a", "b", "c"]
    sharing = share_replicated_seeded({"sk_y": [11, 12], "z": 13}, 2, [1, 2, 3], Q)
    dk = {"sharing": "replicated-seeded", "lst_sid": sids, **seeded_pieces_held(sharing, 1)}
    assert dk["pieces"] == [] and len(dk["seeds"]) == 2  # server 1 is T0 = {1}: seeds only
    share, _, tag = share_for_enrolled(dk, "a", ["a", "b"], 2, Q, fields=("z",))
    assert set(share) == {"sharing", "lst_sid", "z"} and tag == '["replicated", [[2], [3]]]'
    with pytest.raises(FEKeyError):
        share_for_enrolled(dict(dk, seeds=dk["seeds"][:1]), "a", ["a", "b"], 2, Q)
    with pytest.raises(FEKeyError):
        share_for_enrolled(dict(dk, seeds=[], pieces=[]), "a", ["a", "b"], 2, Q)
    with pytest.raises(FEKeyError):
        share_for_enrolled(dict(dk, seeds=[dict(dk["seeds"][0], seed="00")] + dk["seeds"][1:]), "a", ["a", "b"], 2, Q)
    with pytest.raises(FEKeyError):
        share_for_enrolled(dict(dk, layout={"sk_y": "two", "z": None}), "a", ["a", "b"], 2, Q)


# ------------------------------------------------------------------ schemes

def _enrolled_sets(sids, t):
    return [list(e) for size in range(t, len(sids) + 1) for e in combinations(sids, size)]


@pytest.mark.parametrize("s,t", [(3, 2), (4, 3), (3, 3), (2, 1)])
def test_tmife_lwe_seeded_decrypts_and_matches_the_replicated_partials(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": s, "t": t, "lst_nid": ["c0", "c1"], "lst_sid": sids,
           "lwe_n": 16, "bound_x": 6, "bound_y": 6, "sharing": "replicated-seeded"}
    kg = ThresholdMIFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["sharing"] == "replicated-seeded"
    full_pp = _full_pp(pp, tmife_lwe)
    q = gp.mpz(pp["q"])
    x = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    y = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    want = sum(x[n][0] * y[n][0] for n in x)
    cts = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
           for nid in cfg["lst_nid"]}
    dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in sids}
    assert all(len(dk["pieces"]) <= 1 and "sk_y" not in dk for dk in dks.values())
    groups = []
    for enrolled in _enrolled_sets(sids, t):
        group = {}
        for sid in enrolled:
            group[sid] = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
                cts, y, dks[sid], enrolled)
            full = _expanded(dks[sid], q)
            assert ThresholdMIFELWE({"id": sid, "keys": {"pp": full_pp, "dk": full}}).share_decrypt(
                cts, y, full, enrolled) == group[sid]
        groups.append(group)
        assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == want
    assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(groups) == want
    # a seeded key share is refused under replicated pp, and the other way round
    with pytest.raises(FESchemeError):
        ThresholdMIFELWE({"id": sids[0], "keys": {"pp": full_pp, "dk": dks[sids[0]]}}).share_decrypt(
            cts, y, dks[sids[0]], sids[:t])

    ary = {nid: [np.array([[1.0, -2.0], [3.0, 0.0]])] for nid in cfg["lst_nid"]}
    ones = {nid: [1] for nid in cfg["lst_nid"]}
    nd_cts = {nid: ThresholdMIFELWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
              .encrypt_lst_ndarray(ary[nid]) for nid in cfg["lst_nid"]}
    partials = {}
    for sid in sids[-t:]:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": ones})
        partials[sid] = ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).compute_lst_ndarray_ct(
            nd_cts, ones, dk, sids[-t:])
        full = _expanded(dk, q)
        same = ThresholdMIFELWE({"id": sid, "keys": {"pp": full_pp, "dk": full}}).compute_lst_ndarray_ct(
            nd_cts, ones, full, sids[-t:])
        assert [list(a.flat) for a in same] == [list(a.flat) for a in partials[sid]]
    out = ThresholdMIFELWE({"id": "c", "precision": 0, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(partials)
    assert np.array_equal(out[0], np.array([[2.0, -4.0], [6.0, 0.0]]))


@pytest.mark.parametrize("s,t", [(3, 2), (4, 3), (5, 3)])
def test_tmcfe_ring_lwe_seeded_decrypts_and_matches_the_replicated_partials(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": s, "t": t, "lst_nid": ["a", "b", "c"], "lst_sid": sids,
           "ring_n": 16, "bound_x": 4, "bound_y": 4, "sharing": "replicated-seeded"}
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    full_pp = _full_pp(pp, tmcfe_ring)
    q = gp.mpz(pp["q"])
    x = {nid: [[random.randint(-4, 4) for _ in range(5)]] for nid in cfg["lst_nid"]}
    y = {nid: [random.randint(-4, 4)] for nid in cfg["lst_nid"]}
    want = [sum(x[n][0][i] * y[n][0] for n in x) for i in range(5)]
    cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], "L")
           for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "L"}
    dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
    assert [len(dks[sid]["pieces"]) for sid in sids] == [0] * (t - 1) + [1] * (s - t + 1)
    for enrolled in _enrolled_sets(sids, t):
        group = {}
        for sid in enrolled:
            group[sid] = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
                cts, cred, dks[sid], enrolled)
            full = _expanded(dks[sid], q)
            assert ThresholdMCFERingLWE({"id": sid, "keys": {"pp": full_pp, "dk": full}}).share_decrypt(
                cts, cred, full, enrolled) == group[sid]
        assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == want


def test_tmcfe_ring_lwe_seeded_ndarrays_match_the_replicated_partials():
    sids = ["sid_0", "sid_1", "sid_2"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["a", "b"], "lst_sid": sids,
           "ring_n": 16, "bound_x": 40, "bound_y": 4, "sharing": "replicated-seeded"}
    kg = ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    full_pp = _full_pp(pp, tmcfe_ring)
    y = {nid: [2] for nid in cfg["lst_nid"]}
    ary = {nid: [np.arange(40, dtype=float).reshape(5, 8) - 20] for nid in cfg["lst_nid"]}
    nd_cts = {nid: ThresholdMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
              .encrypt_lst_ndarray(ary[nid], label="nd") for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "nd", "n_chunks": ThresholdMCFERingLWE.key_chunks(nd_cts["a"])}
    for enrolled in (["sid_0", "sid_2"], ["sid_1", "sid_2"], sids):
        partials = {}
        for sid in enrolled:
            dk = kg.get_decryption_keys(sid, credentials=cred)
            assert len(dk["layout"]["z_chunks"]) == cred["n_chunks"]
            partials[sid] = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).compute_lst_ndarray_ct(
                nd_cts, cred, dk, enrolled)
            full = _expanded(dk, gp.mpz(pp["q"]))
            assert ThresholdMCFERingLWE({"id": sid, "keys": {"pp": full_pp, "dk": full}}).compute_lst_ndarray_ct(
                nd_cts, cred, full, enrolled) == partials[sid]
        out = ThresholdMCFERingLWE({"id": "c", "precision": 0, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(partials)
        assert np.array_equal(out[0], 4 * (np.arange(40, dtype=float).reshape(5, 8) - 20))


@pytest.mark.parametrize("s,t", [(3, 2), (4, 3)])
def test_tmcfe_lwe_seeded_decrypts_and_matches_the_replicated_partials(s, t):
    sids = ["sid_%d" % i for i in range(s)]
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": s, "t": t, "lst_nid": ["c0", "c1", "c2"], "lst_sid": sids,
           "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2, "label_modulus": 8,
           "sharing": "replicated-seeded"}
    kg = ThresholdMCFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["sharing"] == "replicated-seeded"
    assert pp["threshold_construction"] == "abmpsw20-replicated-seeded-flooding/v1"
    full_pp = _full_pp(pp, tmcfe_lwe)
    x = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    y = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    cts = {nid: ThresholdMCFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], "lbl")
           for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "lbl"}
    dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
    assert all(len(dk["pieces"]) <= 1 and "sk_y" not in dk for dk in dks.values())
    want = sum(x[n][0] * y[n][0] for n in x)
    for enrolled in _enrolled_sets(sids, t):
        group = {}
        for sid in enrolled:
            group[sid] = ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
                cts, cred, dks[sid], enrolled)
            full = _expanded(dks[sid], kg.q)
            assert ThresholdMCFELWE({"id": sid, "keys": {"pp": full_pp, "dk": full}}).share_decrypt(
                cts, cred, full, enrolled) == group[sid]
        assert ThresholdMCFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == want
    # each sharing serves its own key shares only, and the construction must name the sharing
    full = _expanded(dks[sids[0]], kg.q)
    for bad_pp, bad_dk in [(full_pp, dks[sids[0]]), (pp, full),
                           (dict(pp, threshold_construction=tmcfe_lwe._CONSTRUCTION_REPLICATED), dks[sids[0]])]:
        with pytest.raises(FESchemeError):
            ThresholdMCFELWE({"id": sids[0], "keys": {"pp": bad_pp, "dk": bad_dk}}).share_decrypt(
                cts, cred, bad_dk, sids[:t])

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
def test_tmcfe_ring_lwe_seeded_partials_agree_across_backends():
    sids = ["sid_0", "sid_1", "sid_2"]
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["a", "b"], "lst_sid": sids,
           "ring_n": 16, "bound_x": 4, "bound_y": 4, "sharing": "replicated-seeded"}
    y = {nid: [3] for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": "x"}
    for dealer in ("python", "native"):  # dealt with either expansion
        with native_backend.backend(dealer):
            kg = ThresholdMCFERingLWEKeyGenerator(cfg)
            kg.setup()
            pp = kg.get_public_parameters()
            dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
        with native_backend.backend("native"):
            assert native_backend.ring_context(gp.mpz(pp["q"]), pp["ring_n"]) is not None
        cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(
            [[1, -2, 3]], "x") for nid in cfg["lst_nid"]}
        for enrolled in _enrolled_sets(sids, 2):
            group = {}
            for sid in enrolled:
                server = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}})
                with native_backend.backend("python"):
                    a = server.share_decrypt(cts, cred, dks[sid], enrolled)
                server = ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}})  # fresh cache
                with native_backend.backend("native"):
                    b = server.share_decrypt(cts, cred, dks[sid], enrolled)
                assert [str(v) for v in a["w"]] == [str(v) for v in b["w"]]
                group[sid] = b
            assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == [6, -12, 18]


def test_seeded_sharing_has_its_own_parameter_records_and_small_key_shares():
    import os

    def record(kg):
        with open(os.path.join(kg.config_folder, kg._scheme_type, "param.json"), encoding="utf-8") as f:
            return json.load(f)

    sids = ["sid_%d" % i for i in range(5)]
    mife = ThresholdMIFELWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["c0", "c1"], "lwe_n": 16,
                                         "bound_x": 6, "bound_y": 6, "s": 5, "t": 3, "lst_sid": sids,
                                         "sharing": "replicated-seeded"})
    assert record(mife)["construction"] == "acfgu18-abdp15-replicated-seeded-smudging-seeded-a/v1"
    base = {"sec_param": 64, "eta": 1, "n": 2, "lst_nid": ["a", "b"], "ring_n": 16, "bound_x": 4, "bound_y": 4,
            "s": 5, "t": 3, "lst_sid": sids}
    seeded = ThresholdMCFERingLWEKeyGenerator(dict(base, sharing="replicated-seeded"))
    assert record(seeded)["threshold_construction"] == "bmms21-ibmcfe-replicated-seeded-smudging/v1"
    full = ThresholdMCFERingLWEKeyGenerator(dict(base, sharing="replicated"))
    assert seeded.q == full.q and seeded.sigma_f == full.sigma_f  # same parameters
    sizes = {}
    for kg in (seeded, full):
        kg.setup()
        cred = {"fusion_weight": {"a": [1], "b": [1]}, "label": "L", "n_chunks": 2}
        sizes[kg.sharing] = [len(json.dumps(kg.get_decryption_keys(sid, credentials=cred))) for sid in sids]
    # six full pieces per server at (5, 3); seeded: one piece (servers 3..5) or none (1, 2)
    assert max(sizes["replicated-seeded"]) < max(sizes["replicated"]) / 4
    assert max(sizes["replicated-seeded"][:2]) < 1500
