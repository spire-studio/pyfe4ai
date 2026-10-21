"""The threshold DDH key generators deal one sharing per weights, also when
two threads ask for the key shares of the same weights for the first time.

``mife/ddh_threshold`` and ``mcfe/ddh_threshold`` keep the key shares they
issue in ``dict_dk``. Checking, dealing and storing used to be separate
steps: two threads asking at once both dealt, the first returned its
server's share of its own sharing before the second stored another, and
the other servers then got shares of the second sharing. Shares of two
sharings of one key do not reconstruct it: the partial decryptions did not
combine (``None``). The key generators now keep the sharing stored first
(``dict_dk.setdefault``) and issue every key share from it. The tests
replay that interleaving with real threads (the threshold LWE layers: see
tests/review/test_refactor_batch7_review.py; the threshold FH-MCFE:
tests/mcfe/test_fh_multi_ipe_pairing_threshold.py).
"""

from __future__ import annotations

import threading

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDH, ThresholdMCFEDDHKeyGenerator
from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDH, ThresholdMIFEDDHKeyGenerator

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS, SIDS = ["nid_0", "nid_1", "nid_2"], ["sid_0", "sid_1", "sid_2"]
X = {"nid_0": [0.5], "nid_1": [-0.25], "nid_2": [0.125]}
Y = {"nid_0": [2], "nid_1": [1], "nid_2": [3]}
RIGHT = sum(X[n][0] * Y[n][0] for n in NIDS)


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _setup(kind):
    """``(key generator, the name of its dealing method, credentials, partial(sid, dk, enrolled), combine)``."""
    cfg = {"sec_param": 256 if kind == "mife" else 128, "lst_nid": NIDS, "lst_sid": SIDS, "eta": 1, "n": 3,
           "s": 3, "t": 2}
    if kind == "mife":
        kg = ThresholdMIFEDDHKeyGenerator(cfg)
        kg.setup()
        pp = kg.get_public_parameters()
        clients = {n: ThresholdMIFEDDH({"id": n, "precision": 3, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
                   for n in NIDS}
        cts = {n: clients[n].encrypt(X[n]) for n in NIDS}
        creds = {"fusion_weight": Y}

        def partial(sid, dk, enrolled):
            server = ThresholdMIFEDDH({"id": sid, "precision": 3, "keys": {"pp": pp, "dk": dk}})
            return server.share_decrypt(cts, Y, dk, enrolled)

        return kg, "_share_key", creds, partial, clients["nid_0"].combine_decrypt
    kg = ThresholdMCFEDDHKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: ThresholdMCFEDDH({"id": n, "precision": 3, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
               for n in NIDS}
    cts = {n: clients[n].encrypt(X[n], "L") for n in NIDS}
    creds = {"fusion_weight": Y, "label": "L"}

    def partial(sid, dk, enrolled):
        server = ThresholdMCFEDDH({"id": sid, "precision": 3, "keys": {"pp": pp, "dk": dk}})
        return server.share_decrypt(cts, Y, dk, enrolled, "L")

    return kg, "_share_functional_key", creds, partial, clients["nid_0"].combine_decrypt


def first_requests_at_once(kg, dealing: str, creds: dict, first: str, second: str) -> dict:
    """Key shares of servers *first* and *second*, asked for at once by two
    threads: both pass the first-request check and deal, and the first
    returns before the second stores its sharing (a key generator that deals
    once per weights serialises the dealing: the barrier then breaks)."""
    real = getattr(kg, dealing)
    barrier = threading.Barrier(2, timeout=3)
    first_done = threading.Event()

    def racing(*args):
        out = real(*args)
        try:
            barrier.wait()  # both threads have passed the check
        except threading.BrokenBarrierError:
            pass
        if threading.current_thread().name == "second":
            first_done.wait(3)  # the first returns before the second stores
        return out

    setattr(kg, dealing, racing)
    got = {}

    def ask(sid):
        try:
            got[sid] = kg.get_decryption_keys(sid, credentials=creds)
        finally:
            if threading.current_thread().name == "first":
                first_done.set()

    threads = [threading.Thread(target=ask, args=(first,), name="first"),
               threading.Thread(target=ask, args=(second,), name="second")]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    delattr(kg, dealing)
    return got


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["mife", "mcfe"])
def test_concurrent_first_key_requests_issue_one_sharing(kind, backend):
    with native_backend.backend(backend):
        kg, dealing, creds, partial, combine = _setup(kind)
        enrolled = SIDS[:3]
        results = []
        for _ in range(5):
            kg.dict_dk.clear()
            dks = first_requests_at_once(kg, dealing, creds, enrolled[0], enrolled[1])
            assert len(kg.dict_dk) == 1
            dks[enrolled[2]] = kg.get_decryption_keys(enrolled[2], credentials=creds)
            parts = {sid: partial(sid, dks[sid], enrolled) for sid in enrolled}
            results.append(combine(parts))
    assert results == [pytest.approx(RIGHT)] * 5, (
        "{}/ddh_threshold: partial decryptions of key shares asked for at once combine to {} (right: {}): the key "
        "generator handed out shares of two sharings of one key".format(kind, results, RIGHT))
