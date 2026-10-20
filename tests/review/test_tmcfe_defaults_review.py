"""The threshold MCFE defaults (binary, RNS, split key shares) and the ML
adapter's in-process servers (October 2026 code review).

The defaults hold as specified under both profiles, the options of earlier
versions and old data keep working byte-identically, every mix of formats,
parameters and parts fails closed, and the adapter sends no key material.
Before the fixes:

* (availability; fails closed) a threshold aggregation prepared with
  :func:`prepare_aggregation` could not be completed by a pickled or copied
  wrapper -- the copy's servers are made afresh without their weight key
  shares, and :func:`partial_decrypt_gradients` did not give them the weight
  part the prepared label part needs -- nor after the server's weight-part
  store (``WEIGHT_SHARES_KEPT``) dropped it: ``FEKeyError`` in both cases,
  although the wrapper holds the key generator. Copies "for multiprocessing,
  Ray" are a documented use of the wrapper;
* (availability; fails closed) the check that a label part is of the weights'
  current share set ran only when the key generator made the part: a label
  part cached with a sharing that another thread evicted right after
  ``_weight_entry`` returned it was still handed out (servers then hold the new
  weight part and refuse it).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient, prepare_aggregation

NIDS, SIDS = ["nid_0", "nid_1", "nid_2"], ["sid_0", "sid_1", "sid_2"]
WEIGHTS = {"nid_0": 1, "nid_1": 2, "nid_2": 1}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _wrapper():
    cfg = {"eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": 16, "sec_param": 64, "bound_x": 50, "bound_y": 9,
           "s": 3, "t": 2, "lst_sid": SIDS}
    return FESchemeWrapper("mcfe", "ring_lwe_threshold", cfg, precision=1)


def _grads(seed):
    rng = np.random.default_rng(seed)
    return {nid: np.round(rng.uniform(-2, 2, (3, 7)), 1) for nid in NIDS}


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_r5_1_a_prepared_aggregation_completes_in_a_copy_of_the_wrapper(how):
    w = _wrapper()
    grads = _grads(1)
    prepared = prepare_aggregation(w, WEIGHTS, shape=(3, 7), label="a")
    enc = {nid: encrypt_gradient(grads[nid], w, nid=nid, label="a") for nid in NIDS}
    clone = pickle.loads(pickle.dumps(w)) if how == "pickle" else copy.deepcopy(w)
    out = aggregate_gradients(enc, clone, WEIGHTS, label="a", prepared=prepared)
    np.testing.assert_allclose(out, sum(WEIGHTS[nid] * grads[nid] for nid in NIDS), atol=1e-9)


def test_r5_1_a_prepared_aggregation_survives_the_servers_weight_part_eviction():
    w = _wrapper()
    grads = _grads(2)
    prepared = prepare_aggregation(w, WEIGHTS, shape=(3, 7), label="b")
    for i in range(tm.ThresholdMCFERingLWE.WEIGHT_SHARES_KEPT + 1):  # other weights in other rounds
        other = {"nid_0": 1 + i % 3, "nid_1": 1 + (i // 3) % 3, "nid_2": 4 + (i // 9) % 3}
        g = _grads(100 + i)
        aggregate_gradients({nid: encrypt_gradient(g[nid], w, nid=nid, label="o%d" % i) for nid in NIDS}, w, other,
                            label="o%d" % i)
    enc = {nid: encrypt_gradient(grads[nid], w, nid=nid, label="b") for nid in NIDS}
    out = aggregate_gradients(enc, w, WEIGHTS, label="b", prepared=prepared)
    np.testing.assert_allclose(out, sum(WEIGHTS[nid] * grads[nid] for nid in NIDS), atol=1e-9)


def test_r5_2_a_cached_label_part_is_of_the_current_share_set(monkeypatch):
    monkeypatch.setattr(tm, "WEIGHT_SHARINGS_KEPT", 1)
    nids, sids = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
    kg = tm.ThresholdMCFERingLWEKeyGenerator(dict(n=3, lst_nid=nids, s=3, t=3, lst_sid=sids, eta=1, ring_n=32,
                                                  sec_param=64, bound_x=100, bound_y=3, stat_param=64))
    kg.setup()
    weights, other = {n: [1] for n in nids}, {n: [2] for n in nids}
    creds = {"fusion_weight": weights, "label": "L"}
    kg.get_decryption_keys("s0", credentials=creds)  # cached with the first sharing
    real, armed = kg._weight_entry, [True]

    def racing(fw):
        entry = real(fw)
        if armed:
            armed.clear()
            real(other)  # meanwhile another thread: other weights evict this sharing ...
            real(weights)  # ... and these weights return with a new share set
        return entry

    kg._weight_entry = racing
    try:
        dk = kg.get_decryption_keys("s1", credentials=creds)
    finally:
        del kg._weight_entry
    current = kg.get_weight_key_share("s1", credentials={"fusion_weight": weights})
    assert dk["share_set_id"] == current["share_set_id"], "a label part of an evicted share set was handed out"
