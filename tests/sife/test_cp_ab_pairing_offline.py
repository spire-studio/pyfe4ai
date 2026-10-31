"""Offline / online encryption of the attribute-based IPFE (sife/cp_ab_pairing): the ciphertext of ``precompute`` and
``encrypt(x, precomputed=...)`` is the one ``encrypt`` gives for the same randomness, and a precomputation is used once.
"""

import copy
import os
import pickle
import threading

import pytest

from tests.pairing_group import skip_unless_available  # noqa: E402

skip_unless_available()

from tests.sife.conftest import ETA  # noqa: E402
from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.policy import parse
from pyfe4ai.primitives.pairing.pairing_backend import ZR
from pyfe4ai.schemes.sife import cp_ab_pairing
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing

pytestmark = pytest.mark.usefixtures("own_parameter_cache")

POLICY = "and(org:A, role:auditor)"
LABELS = ["org:A", "role:auditor"]
X = [3, -1, 0, 7, 2]
Y = [1, 2, 3, -4, 5]
VALUE = sum(a * b for a, b in zip(X, Y))


def key(kg, attributes=LABELS, y=Y):
    return kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": attributes})


@pytest.mark.parametrize("policy, labels", [
    (POLICY, LABELS), ("org:A", ["org:A"]), ("or(org:A, org:B)", ["org:B"]),
    ("2of(org:A, org:B, role:auditor)", ["org:B", "role:auditor"]),
])
def test_an_online_ciphertext_decrypts(make, policy, labels):
    kg, crypto = make()
    ct = crypto.encrypt(X, precomputed=crypto.precompute(policy))
    assert crypto.decrypt(ct, key(kg, labels)) == VALUE


def test_the_policy_may_be_a_formula(make):
    kg, crypto = make()
    assert crypto.decrypt(crypto.encrypt(X, precomputed=crypto.precompute(parse(POLICY))), key(kg)) == VALUE


@pytest.mark.parametrize("x", [[0] * ETA, [10] * ETA, [-10] * ETA, [10, -10, 0, 1, -1]])
def test_an_online_ciphertext_takes_every_plaintext_in_bounds(make, x):
    kg, crypto = make()
    expected = sum(a * b for a, b in zip(x, Y))
    assert crypto.decrypt(crypto.encrypt(x, precomputed=crypto.precompute(POLICY)), key(kg)) == expected


@pytest.mark.parametrize("threads", [1, 4])
def test_the_same_randomness_gives_the_ciphertext_of_encrypt(make, monkeypatch, threads):
    _, crypto = make()
    monkeypatch.setattr(cp_ab_pairing, "random_nonzero_zr", lambda group: group.init(ZR, 1234567891011))
    native_backend.set_threads(threads)
    try:
        assert crypto.encrypt(X, precomputed=crypto.precompute(POLICY)) == crypto.encrypt(X, policy=POLICY)
    finally:
        native_backend.set_threads(1)


def test_each_precomputation_has_its_own_randomness(make):
    _, crypto = make()
    first, second = (crypto.encrypt(X, precomputed=crypto.precompute(POLICY)) for _ in range(2))
    assert first["c1"] != second["c1"] and first["c3"] != second["c3"]


def test_a_precomputation_is_used_once(make):
    kg, crypto = make()
    pre = crypto.precompute(POLICY)
    assert crypto.decrypt(crypto.encrypt(X, precomputed=pre), key(kg)) == VALUE
    with pytest.raises(FEValidationError, match="used once"):
        crypto.encrypt(X, precomputed=pre)


def test_use_drops_the_secret_parts(make):
    _, crypto = make()
    pre = crypto.precompute(POLICY)
    crypto.encrypt(X, precomputed=pre)
    assert pre.used and pre.parts is None and pre.masks is None


def test_the_repr_holds_nothing_of_the_randomness(make):
    _, crypto = make()
    pre = crypto.precompute(POLICY)
    assert repr(pre) == "CPABPrecomputation(used=False)"


@pytest.mark.parametrize("bad", [
    {"lst_pt": [1, 2]}, {"lst_pt": [99] * ETA}, {"lst_pt": X, "policy": POLICY}, {"lst_pt": ["1"] * ETA},
], ids=["short", "over-the-bound", "policy-as-well", "not-integers"])
def test_refused_input_does_not_consume_it(make, bad):
    kg, crypto = make()
    pre = crypto.precompute(POLICY)
    with pytest.raises(FEValidationError):
        crypto.encrypt(precomputed=pre, **bad)
    assert not pre.used
    assert crypto.decrypt(crypto.encrypt(X, precomputed=pre), key(kg)) == VALUE


def test_the_precomputation_of_another_object_is_refused(make):
    kg, crypto = make()
    other = SIFECPABPairing({"keys": {"pp": kg.get_public_parameters()}})
    pre = other.precompute(POLICY)
    with pytest.raises(FEValidationError, match="another"):
        crypto.encrypt(X, precomputed=pre)
    assert not pre.used


@pytest.mark.parametrize("bad", [{"masks": []}, object(), 0, ""], ids=["dict", "object", "zero", "empty-text"])
def test_anything_else_is_refused_as_a_precomputation(make, bad):
    _, crypto = make()
    with pytest.raises(FEValidationError, match="another"):
        crypto.encrypt(X, precomputed=bad)


@pytest.mark.parametrize("policy, match", [
    (None, "must be a Formula"), (7, "must be a Formula"), ("org:Z", "not in the universe"),
    ("and(org:A, org:B, org:C, role:auditor)", "columns"),
])
def test_precompute_refuses_the_policies_encrypt_refuses(make, policy, match):
    _, crypto = make(max_columns=3)
    with pytest.raises(FEValidationError, match=match):
        crypto.encrypt(X, policy=policy)
    with pytest.raises(FEValidationError, match=match):
        crypto.precompute(policy)


def test_copies_and_pickles_are_refused(make):
    _, crypto = make()
    pre = crypto.precompute(POLICY)
    for clone in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            clone(pre)


def test_one_of_many_threads_gets_it(make):
    _, crypto = make()
    pre = crypto.precompute(POLICY)
    results, barrier = [], threading.Barrier(4)

    def run():
        barrier.wait()
        try:
            crypto.encrypt(X, precomputed=pre)
            results.append("ok")
        except FEValidationError:
            results.append("refused")

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert sorted(results) == ["ok", "refused", "refused", "refused"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_cannot_use_it(make):
    kg, crypto = make()
    pre = crypto.precompute(POLICY)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            crypto.encrypt(X, precomputed=pre)
            os.write(write, b"used")
        except FEValidationError:
            os.write(write, b"refused" if pre.used else b"refused, not invalidated")
        except BaseException as exc:  # noqa: BLE001
            os.write(write, type(exc).__name__.encode())
        os._exit(0)
    os.close(write)
    outcome = os.read(read, 100).decode()
    os.close(read)
    os.waitpid(pid, 0)
    assert outcome == "refused"
    assert crypto.decrypt(crypto.encrypt(X, precomputed=pre), key(kg)) == VALUE    # still the parent's, once
