"""One-time use of the offline randomness (October 2026 code review).

``OfflineRandomMixin`` (SIFE-/MIFE-Paillier, MIFE-DDH) stores ``r``-dependent
parts per element; ``_encrypt_precomputed`` marked the object used and dropped
its reference to the parts. A shallow copy taken before the first use (or the
copy a ``fork()`` makes) still had ``used = False`` and the same parts, so a
second gradient was encrypted with the same ``r``: for SIFE-Paillier
``ct1 / ct1' = (1 + N)^(x - x') mod N^2`` gives the difference without any key.
Precomputations are now claimed atomically, refuse copies and pickles, and
are invalidated in a forked child.
"""

import copy

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.sife import paillier as sife_paillier
from pyfe4ai.core.exceptions import FEValidationError


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_a_copied_precomputation_cannot_encrypt_twice():
    kg = sife_paillier.SIFEPaillierKeyGenerator({"eta": 1, "bit_length": 256, "bound_x": 20, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = sife_paillier.SIFEPaillier({"keys": {"pp": pp, "sk": kg.get_private_keys()}, "precision": 0})
    pre = enc.precompute([(1,)])
    try:
        twin = copy.copy(pre)
    except TypeError:
        return  # the copy refused: the fix (copies are refused outright)
    first = enc.encrypt_lst_ndarray([np.array([7])], precomputed=pre)[0][0]
    try:
        second = enc.encrypt_lst_ndarray([np.array([-5])], precomputed=twin)[0][0]
    except (FEValidationError, TypeError):
        return  # refused: the fix
    n, n2 = gp.mpz(pp["n"]), gp.mpz(pp["n_square"])
    ratio = gp.mpz(first["ct1"][0]) * gp.invert(gp.mpz(second["ct1"][0]), n2) % n2
    assert first["ct0"] != second["ct0"], "same r twice: x - x' = {}".format(int((ratio - 1) // n))


def test_two_threads_cannot_use_one_precomputation():
    import threading

    kg = sife_paillier.SIFEPaillierKeyGenerator({"eta": 1, "bit_length": 256, "bound_x": 20, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = sife_paillier.SIFEPaillier({"keys": {"pp": pp, "sk": kg.get_private_keys()}, "precision": 0})
    size = 20000
    rng = np.random.default_rng(1)
    pre = enc.precompute([(size,)])
    xs = {"A": rng.integers(-20, 21, size), "B": rng.integers(-20, 21, size)}
    out, bar = {}, threading.Barrier(2)

    def run(name):
        bar.wait()
        try:
            out[name] = enc.encrypt_lst_ndarray([xs[name]], precomputed=pre)[0]
        except FEValidationError:
            pass

    threads = [threading.Thread(target=run, args=(n,)) for n in xs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if len(out) < 2:
        return
    same = sum(out["A"][i]["ct0"] == out["B"][i]["ct0"] for i in range(size))
    pytest.fail("one precomputation encrypted two gradients: {} of {} elements share r".format(same, size))
