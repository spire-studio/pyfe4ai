"""ML adapter: one ciphertext per client and label under threads (October 2026
code review; the rule dates from v0.3.0).

``encrypt_gradient`` refuses a second ciphertext of a client under one label,
but used to check ``(nid, label)`` before encrypting and record it afterwards.
Two threads encrypting for the same client and label both succeeded (the
native backend releases the GIL during encryption, which widens the window),
and an aggregator that aggregates the label twice with the same weights learns
``gA - gB``. The pair is now reserved before encrypting.
"""

import threading

import numpy as np
import pytest

from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient

NIDS = ["c0", "c1", "c2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_one_ciphertext_per_client_and_label_under_threads():
    w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": 1024, "sec_param": 128,
                                             "bound_x": 1000, "bound_y": 1}, precision=2)
    shape = (300, 1024)
    rng = np.random.default_rng(0)
    g_a, g_b = rng.uniform(-1, 1, shape).round(2), rng.uniform(-1, 1, shape).round(2)
    out, bar = {}, threading.Barrier(2)

    def run(name, g):
        bar.wait()
        try:
            out[name] = encrypt_gradient(g, w, nid="c0", label="round-7")
        except ValueError:
            pass

    threads = [threading.Thread(target=run, args=("A", g_a)), threading.Thread(target=run, args=("B", g_b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if len(out) < 2:
        return
    others = {n: encrypt_gradient(rng.uniform(-1, 1, shape).round(2), w, nid=n, label="round-7") for n in NIDS[1:]}
    weights = {n: 1 for n in NIDS}
    s_a = aggregate_gradients(dict(others, c0=out["A"]), w, weights, label="round-7")
    s_b = aggregate_gradients(dict(others, c0=out["B"]), w, weights, label="round-7")
    assert not np.allclose(s_a - s_b, g_a - g_b), "two ciphertexts of c0 under one label: gA - gB leaks"
    pytest.fail("two concurrent encrypt_gradient calls for (c0, round-7) both succeeded")
