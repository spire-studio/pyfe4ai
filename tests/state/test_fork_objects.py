"""Objects with a lock of their own, in a forked child.

A :class:`~pyfe4ai.schemes.threshold.kl.PartialDecryptionBudget` counts in
the process that made it only: copied into a forked child it refuses to
count (the parent and the child would otherwise each count from the copied
``used`` and together exceed ``max_partial_decryptions``, the bound of the
KL flooding guarantee), and so does a ``"kl"`` server counting against it.
The locks of a budget and of a :class:`~pyfe4ai.ml.FESchemeWrapper` are new
in the child (:func:`pyfe4ai.core.forksafe.track`): one that another thread
of the parent held at the fork no longer blocks the child. Each child runs
under an alarm, so that a child that blocks fails the test instead of
hanging it.
"""

from __future__ import annotations

import signal

import numpy as np
import pytest

from pyfe4ai.core.exceptions import FESchemeError
from pyfe4ai.ml import FESchemeWrapper, encrypt_gradient
from pyfe4ai.schemes.threshold.kl import PartialDecryptionBudget
from tests.state.test_forksafe import fork_warning, held_by_another_thread, in_child, needs_fork

#: Seconds a child may take before SIGALRM ends it (a lock it waits for forever).
ALARM = 10


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def alarmed(check):
    """*check*, ended by SIGALRM after :data:`ALARM` seconds (in the child)."""
    def run():
        signal.alarm(ALARM)
        try:
            return check()
        finally:
            signal.alarm(0)

    return run


@needs_fork
@fork_warning
def test_a_budget_refuses_to_count_in_a_forked_child():
    budget = PartialDecryptionBudget(5, used=2)

    def child():
        for call in (budget.consume, budget.check):
            with pytest.raises(FESchemeError, match="process that made it"):
                call()
        assert budget.used == 2 and budget.remaining == 3  # nothing counted here

    assert in_child(alarmed(child)) == (0, "")
    budget.consume(3)  # the parent counts on
    assert budget.remaining == 0
    with pytest.raises(FESchemeError, match="limit reached"):
        budget.consume()


@needs_fork
@fork_warning
def test_a_budget_lock_held_at_the_fork_is_new_in_the_child():
    budget = PartialDecryptionBudget(5)
    with held_by_another_thread(budget, "_lock") as old:

        def child():
            assert budget._lock is not old, "the parent's lock"
            assert budget.remaining == 5  # does not block
            with pytest.raises(FESchemeError, match="process that made it"):
                budget.consume()

        assert in_child(alarmed(child)) == (0, "")
        assert budget._lock is old
    budget.consume()
    assert budget.used == 1


@needs_fork
@fork_warning
def test_a_kl_server_refuses_to_answer_against_a_budget_copied_into_a_child():
    from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm

    nids, sids = ["c0", "c1"], ["s0", "s1"]
    weights = {"c0": [1], "c1": [2]}
    kg = tm.ThresholdMCFERingLWEKeyGenerator({"n": 2, "lst_nid": nids, "s": 2, "t": 2, "lst_sid": sids, "eta": 1,
                                              "ring_n": 16, "sec_param": 64, "bound_x": 10, "bound_y": 3,
                                              "flooding_accounting": "kl", "max_partial_decryptions": 4})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: tm.ThresholdMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
           .encrypt([[1, 2]], "L") for n in nids}
    creds = {"fusion_weight": weights, "label": "L", "key_part": "full"}
    budget = PartialDecryptionBudget(4)
    server = tm.ThresholdMCFERingLWE({"id": "s0", "precision": 0, "partial_budget": budget, "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("s0", credentials=creds)
    server.share_decrypt(cts, creds, dk, sids)
    assert budget.used == 1

    def child():
        with pytest.raises(FESchemeError, match="process that made it"):
            server.share_decrypt(cts, creds, dk, sids)
        assert budget.used == 1

    assert in_child(alarmed(child)) == (0, "")
    server.share_decrypt(cts, creds, dk, sids)
    assert budget.used == 2


@needs_fork
@fork_warning
def test_a_wrapper_label_lock_held_at_the_fork_is_new_in_the_child():
    wrapper = FESchemeWrapper("mcfe", "ring_lwe", {"n": 1, "lst_nid": ["c0"], "eta": 1, "ring_n": 16,
                                                   "sec_param": 64, "bound_x": 100, "bound_y": 3}, precision=1)
    encrypt_gradient(np.ones(3), wrapper, nid="c0", label="L0")
    with held_by_another_thread(wrapper, "_label_lock") as old:

        def child():
            assert wrapper._label_lock is not old, "the parent's lock"
            encrypt_gradient(np.ones(3), wrapper, nid="c0", label="L1")  # takes the label lock
            with pytest.raises(ValueError, match="already encrypted"):
                encrypt_gradient(np.ones(3), wrapper, nid="c0", label="L0")

        assert in_child(alarmed(child)) == (0, "")
        assert wrapper._label_lock is old
    encrypt_gradient(np.ones(3), wrapper, nid="c0", label="L1")  # the child's labels stay in the child


@needs_fork
@fork_warning
def test_copies_of_a_wrapper_get_a_new_lock_in_the_child_too():
    import copy
    import pickle

    wrapper = FESchemeWrapper("mcfe", "ring_lwe", {"n": 1, "lst_nid": ["c0"], "eta": 1, "ring_n": 16,
                                                   "sec_param": 64, "bound_x": 100, "bound_y": 3}, precision=1)
    for twin in (copy.copy(wrapper), copy.deepcopy(wrapper), pickle.loads(pickle.dumps(wrapper))):
        with held_by_another_thread(twin, "_label_lock") as old:

            def child(twin=twin, old=old):
                assert twin._label_lock is not old
                assert twin._label_lock.acquire(timeout=2)

            assert in_child(alarmed(child)) == (0, "")
