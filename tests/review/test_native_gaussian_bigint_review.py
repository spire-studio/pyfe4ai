"""Review 2026-10 (perf/native-gaussian-bigint 91d0369): native discrete_gaussian_bigint.

No wrong sample was found (17.6M differential samples, crafted near-half and
tie cases, no 53-bit constant as in finding M4); the first test keeps a
spot check of the bit-identity. Fail on the reviewed commit:

* an unhashable sigma (``gmpy2.xmpz``) raises ``TypeError`` in the
  ``lru_cache`` of ``_bigint_width``, where the reference samples;
* a sigma with an exponent below ``-2^20`` passes the Python gate and the
  extension raises ``ValueError``, where the reference returns zeros;
* the ``XofRandom`` counter is read before and advanced after the call, which
  releases the GIL: another thread drawing from the same ``XofRandom`` during
  the call gets the same SHAKE blocks as the native samples (located by value;
  fixed on cc37d18, which claims the calls before the extension runs).
"""

import os
import threading

import gmpy2 as gp
import pytest

from pyfe4ai.utils import native_backend as nb
from pyfe4ai.utils import sampling_utils as su

pytestmark = pytest.mark.skipif(nb.ext() is None or not hasattr(nb.ext(), "gauss_bigint"),
                                reason="native gauss_bigint not built")


@pytest.mark.parametrize("sigma", [3.19, 2 ** 49 + 12345, gp.mpz(2) ** 200 * 3 + 1, 2 ** 447 - 1, 0.75])
def test_native_samples_are_the_reference(sigma):
    seed = os.urandom(32)
    with nb.backend("native"):
        a = su.discrete_gaussian_bigint(sigma, 2000, rng=su.XofRandom(seed), sampler="rounded")
    with nb.backend("python"):
        b = su.discrete_gaussian_bigint(sigma, 2000, rng=su.XofRandom(seed), sampler="rounded")
    assert a == b


@pytest.mark.parametrize("sigma", [gp.xmpz(1000), gp.mpfr("1e-400000")], ids=["xmpz", "tiny"])
def test_native_path_accepts_what_the_reference_accepts(sigma):
    seed = b"\x01" * 32
    with nb.backend("python"):
        want = su.discrete_gaussian_bigint(sigma, 5, rng=su.XofRandom(seed), sampler="rounded")
    with nb.backend("native"):
        got = su.discrete_gaussian_bigint(sigma, 5, rng=su.XofRandom(seed), sampler="rounded")
    assert got == want


def test_concurrent_draws_do_not_reuse_the_native_range():
    """The blocks a concurrent thread draws from the same XofRandom must not be
    among the native call's blocks. The thread's blocks are located by value (the
    first 8 bytes of SHAKE-256(head || counter) are its getrandbits(64)); the
    native call's range is the xof_counter the extension receives. (A first
    version anchored the window at the counter before the race and so counted
    the thread's draws before the claim: it could not pass on correct code.)"""
    import hashlib

    seed, count = b"\x02" * 32, 6000
    rng = su.XofRandom(seed)
    stop, outs, got = threading.Event(), [], {}
    real = nb._ext

    class Spy:
        def __getattr__(self, name):
            f = getattr(real, name)
            if name != "gauss_bigint":
                return f

            def g(*a, **kw):
                got["xc"] = kw.get("xof_counter")
                return f(*a, **kw)
            return g

    def other():
        while not stop.is_set():
            outs.append(rng.getrandbits(64))

    t = threading.Thread(target=other)
    nb._ext = Spy()
    try:
        with nb.backend("native"):
            t.start()
            # the MPFR path's native kernel, whatever PYFE4AI_GAUSSIAN_SAMPLER selects
            su.discrete_gaussian_bigint(2 ** 100, count, rng=rng, sampler="rounded")
            stop.set()
            t.join()
    finally:
        nb._ext = real
    head = len(seed).to_bytes(8, "big") + seed
    block = {int.from_bytes(hashlib.shake_256(head + c.to_bytes(8, "big")).digest(8), "big"): c
             for c in range(rng._counter + 2 * count)}
    used = [block[v] for v in outs]
    xc = got["xc"]
    assert len(set(used)) == len(used), "the other thread drew a block twice"
    reused = [c for c in used if xc <= c < xc + 2 * count]
    assert not reused, "{} blocks of the native call were also drawn by another thread".format(len(reused))
