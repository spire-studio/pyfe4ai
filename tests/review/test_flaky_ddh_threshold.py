"""Review 2026-10: the intermittent failure of the full suite (perf/phase2 34d2063, also main).

``tests/mcfe/test_ddh_threshold.py::test_entire_process_lst_ndarray`` encrypts
random floats at precision 3 and compares the decryption with the inner
product of the *unquantised* floats at tolerance 1e-2. The rounding error
``sum_i y_i delta_i`` (``|delta_i| <= 5e-4``, ``y_i`` in 1..10, three clients)
reaches 0.015, so the test fails in about 0.35 % of runs (Monte Carlo). The
draws below are those of a failing shuffled run (seed 111): ``y = [5, 10, 8]``
and the three floats of check point ``[0][(0, 1)]`` (error 0.0112). Not a
cryptographic defect: the decryption is exact for the quantised inputs.
"""

import random

import numpy as np

from tests.mcfe import test_ddh_threshold as original

X01 = [0.06850995970323248, 0.17850367864086247, 0.13452971734277064]


def test_original_test_fails_on_the_recorded_draws(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = {"x": 0, "y": iter([5, 10, 8])}

    def random_sample(shape):
        ary = np.full(shape, 0.25)
        if calls["x"] % 3 == 0:  # layer 0 of the next client
            ary[(0, 1)] = X01[calls["x"] // 3]
        calls["x"] += 1
        return ary

    monkeypatch.setattr(np.random, "random_sample", random_sample)
    monkeypatch.setattr(random, "randint", lambda a, b: next(calls["y"]))
    config = {"sec_param": 128, "lst_nid": ["nid_0", "nid_1", "nid_2"], "lst_sid": ["sid_0", "sid_1"],
              "eta": 1, "n": 3, "s": 2, "t": 2}
    try:
        original.test_entire_process_lst_ndarray(config)
    except AssertionError as exc:
        raise AssertionError("tolerance 1e-2 below the quantisation error of precision 3") from exc
