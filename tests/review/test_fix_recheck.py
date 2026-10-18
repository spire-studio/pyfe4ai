"""Re-check 2026-10 of fix/crypto-review-2026-10 (0ca10e8): regression of the M6 fix.

``FESchemeWrapper`` now holds a ``threading.Lock`` (``_label_lock``), so it
can no longer be pickled or deep-copied (multiprocessing, Ray/cloudpickle);
on 34d2063 both worked. Fails on the re-checked commit. A fix should also
decide whether a copy shares the (nid, label) registry or refuses it.
"""

import copy
import pickle

import pytest

from pyfe4ai.utils.ml_adapter import FESchemeWrapper


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("clone", [lambda w: pickle.loads(pickle.dumps(w)), copy.deepcopy], ids=["pickle", "deepcopy"])
def test_wrapper_can_still_be_pickled_and_copied(clone):
    w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 2, "lst_nid": ["a", "b"], "ring_n": 64, "sec_param": 64,
                                             "bound_x": 100, "bound_y": 1}, precision=2)
    clone(w)
