"""ML adapter wrappers stay picklable and copyable (October 2026 code review).

The fix of the ``(nid, label)`` race (test_adapter_label_race.py) gave
``FESchemeWrapper`` a ``threading.Lock`` (``_label_lock``), so it could no
longer be pickled or deep-copied (multiprocessing, Ray/cloudpickle), which
worked before. A copy now carries the pairs used so far and refuses them.
"""

import copy
import pickle

import pytest

from pyfe4ai.ml import FESchemeWrapper


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("clone", [lambda w: pickle.loads(pickle.dumps(w)), copy.deepcopy], ids=["pickle", "deepcopy"])
def test_wrapper_can_still_be_pickled_and_copied(clone):
    w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 2, "lst_nid": ["a", "b"], "ring_n": 64, "sec_param": 64,
                                             "bound_x": 100, "bound_y": 1}, precision=2)
    clone(w)
