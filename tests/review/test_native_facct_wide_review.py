"""Review 2026-10 (perf/native-facct-wide d898446): native FACCT for narrow and wide widths.

No wrong sample was found (bit-identical to the Python reference on 349 widths
with the same XofRandom stream, crafted near-threshold bits, special cases;
the first test keeps a spot check). Fail on the reviewed commit:

* the native XofRandom bit source counts calls in a u64 that wraps: from a
  counter near 2^64 the reference raises OverflowError, the native draw
  re-reads the blocks of counters 0, 1, ... and rewinds the XofRandom;
* XofRandom._claim_dynamic holds the global, non-reentrant _XOF_LOCK across
  the whole native draw and the Python code around it: a signal handler (or
  finalizer) that draws from any XofRandom meanwhile waits for the lock
  forever (here: unbounded re-entry, RecursionError).
"""

import os
import subprocess
import sys
import textwrap

import gmpy2 as gp
import pytest

from pyfe4ai.utils import native_backend as nb
from pyfe4ai.utils import sampling_utils as su

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(nb.ext() is None, reason="native extension not built")


@pytest.mark.parametrize("sigma", [2.0 ** 52 * 3, gp.mpz(2) ** 63 + 5, gp.mpz(2) ** 200 * 7 + 1, 3.19])
def test_native_facct_samples_are_the_reference(sigma):
    seed = os.urandom(32)
    with nb.backend("native"):
        r1 = su.XofRandom(seed)
        a = su.discrete_gaussian_facct(sigma, 300, rng=r1)
    with nb.backend("python"):
        r2 = su.XofRandom(seed)
        b = su.discrete_gaussian_facct(sigma, 300, rng=r2)
    assert a == b and r1._counter == r2._counter


def test_xof_counter_near_2_64_is_not_wrapped():
    sigma = gp.mpz(2) ** 100
    start = 2 ** 64 - 3
    rng = su.XofRandom(b"\x05" * 32)
    rng._counter = start
    with nb.backend("native"):
        try:
            su.discrete_gaussian_facct(sigma, 2, rng=rng)
        except OverflowError:
            return  # as the reference
    assert rng._counter >= start, "the native draw wrapped the counter to {}".format(rng._counter)


_SIGNAL = textwrap.dedent(
    """
    import signal, sys, time
    from pyfe4ai.utils import native_backend as nb, sampling_utils as su
    other, rng = su.XofRandom(b"o" * 32), su.XofRandom(b"r" * 32)
    def handler(*args):
        other.getrandbits(8)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.0005, 0.0005)
    t0 = time.time()
    with nb.backend("native"):
        while time.time() - t0 < 5:
            su.discrete_gaussian_facct(1e30, 1, rng=rng)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print("ok", flush=True)
    """
)


@pytest.mark.skipif(not hasattr(__import__("signal"), "setitimer"), reason="needs setitimer")
def test_signal_handler_drawing_from_an_xof_random_does_not_hang():
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        res = subprocess.run([sys.executable, "-c", _SIGNAL], capture_output=True, text=True, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("a signal handler drawing from an XofRandom hung on _XOF_LOCK")
    assert res.returncode == 0 and res.stdout.strip() == "ok", (res.returncode, res.stderr[-300:])
