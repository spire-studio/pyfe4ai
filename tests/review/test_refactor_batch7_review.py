"""Review of refactor batch 7 (global state, fork safety, thread-safety contract; 0bb8e11..8c37871).

Medium (integrity; not introduced by batch 7, the open item O5): the key generators of the threshold LWE
layers (``mife/lwe_threshold``, ``mcfe/lwe_threshold``) keep the key shares they issue in ``dict_dk``
without a lock -- check, deal, store, read. Two threads asking for the same weights for the first time
both deal; the first returns its server's share of its sharing before the second stores its own, and
every later server gets a share of the second sharing. These key shares carry no sharing id, so a
combiner cannot refuse the mix: shares of two sharings of one key reconstruct a wrong value, which the
decoding mostly accepts (tMCFE-LWE: 99-100 % of the trials decode to a wrong value inside the bound;
tMIFE-LWE: 16-25 %, the others None). ``docs/threading.rst`` asks to issue these key shares from one
thread, but the outcome of doing otherwise is a silently wrong aggregate, not a refusal (the ``MixedSharingsError``
of the threshold MCFE over Ring-LWE does not exist here).

The test replays the interleaving with real threads: both pass the check, both deal, the first returns
before the second stores. A key generator that keeps one sharing per weights (``dict_dk.setdefault``
and reading the stored entry, or a lock around check, deal and store) passes it.
"""

from __future__ import annotations

import threading

import pytest

from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
X = {"c0": [3], "c1": [-2], "c2": [5]}
W = {"c0": [2], "c1": [1], "c2": [-1]}
RIGHT = sum(X[n][0] * W[n][0] for n in NIDS)


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _setup(kind, t, sharing):
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": t, "lst_nid": NIDS, "lst_sid": SIDS, "lwe_n": 16,
           "bound_x": 6, "bound_y": 6, "sharing": sharing}
    if kind == "mcfe":
        cfg.update(bound_u=2, label_modulus=8)
    kg_cls, crypto = (ThresholdMIFELWEKeyGenerator, ThresholdMIFELWE) if kind == "mife" else (
        ThresholdMCFELWEKeyGenerator, ThresholdMCFELWE)
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: crypto({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in NIDS}
    cts = {n: (c.encrypt(X[n]) if kind == "mife" else c.encrypt(X[n], "L")) for n, c in clients.items()}
    creds = {"fusion_weight": W} if kind == "mife" else {"fusion_weight": W, "label": "L"}
    return kg, pp, crypto, cts, creds, (W if kind == "mife" else creds)


def _first_requests_at_once(kg, creds, first, second):
    """Key shares of *first* and *second*, asked for at once by two threads, in the interleaving above."""
    real = kg._share_key
    barrier = threading.Barrier(2, timeout=3)
    first_done = threading.Event()

    def racing(key):
        out = real(key)
        try:
            barrier.wait()  # both threads have passed the check
        except threading.BrokenBarrierError:  # a key generator that serialises the dealing
            pass
        if threading.current_thread().name == "second":
            first_done.wait(3)  # the first returns before the second stores
        return out

    kg._share_key = racing
    got = {}

    def ask(sid):
        got[sid] = kg.get_decryption_keys(sid, credentials=creds)
        if threading.current_thread().name == "first":
            first_done.set()

    threads = [threading.Thread(target=ask, args=(first,), name="first"),
               threading.Thread(target=ask, args=(second,), name="second")]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    del kg._share_key
    return got


@pytest.mark.parametrize("kind", ["mife", "mcfe"])
@pytest.mark.parametrize("t,sharing", [(3, "shamir"), (2, "replicated")])
def test_o5_concurrent_first_key_requests_do_not_mix_two_sharings(kind, t, sharing):
    kg, pp, crypto, cts, creds, arg = _setup(kind, t, sharing)
    enrolled = SIDS[:t]
    results = []
    for _ in range(5):
        kg.dict_dk.clear()
        dks = _first_requests_at_once(kg, creds, enrolled[0], enrolled[1])
        for sid in enrolled[2:]:
            dks[sid] = kg.get_decryption_keys(sid, credentials=creds)
        parts = {sid: crypto({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(cts, arg, dks[sid], enrolled)
                 for sid in enrolled}
        results.append(crypto({"id": "agg", "keys": {"pp": pp}}).combine_decrypt(parts))
    assert results == [RIGHT] * 5, (
        "{}-lwe {}: partial decryptions of key shares asked for at once combine to {} (right: {}); the key "
        "generator handed out shares of two sharings of one key".format(kind, sharing, results, RIGHT))
