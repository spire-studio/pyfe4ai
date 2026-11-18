"""Multi-threaded native kernels run in a forked child after the parent ran them with several threads.

On macOS the standard library of Rust parks a thread on a dispatch semaphore whose Mach port does not survive
``fork()``. ``pyfe4ai_core::parallel_for`` used to wait for its workers in ``std::thread::scope``, which parks the
calling (Python) thread whenever the workers finish after it; the next such wait of that thread in a forked child
aborted the process (SIGTRAP, "BUG IN CLIENT OF LIBDISPATCH: Use-after-free of dispatch_semaphore_t", "crashed on
child side of fork pre-exec"). ``parallel_for`` now delegates to ``parallel_for_joined``: the caller joins every
worker it spawned and never parks.

Each case runs in a fresh interpreter, so that a crash cannot take the test run with it. The parent runs one kernel
with 4 threads, forks four times, and each child runs both kernels with 4 threads for several rounds and compares
their results with the parent's; the parent goes on with 4 threads after each fork and reports how each child ended.
The kernels are ``dense_dots`` (dense-LWE label dots) and ``r255_mcfe_encrypt`` (ristretto255 MCFE, per element),
which reproduced the crash. Skipped without ``os.fork`` or the extension.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pyfe4ai import backend as native_backend

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(not native_backend.available(), reason="native extension not built"),
    pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is not available"),
]

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent("""
    import os, sys
    from pyfe4ai import backend

    backend.set_threads(4)
    ops, ext = backend.ops(), backend.ext()
    MATERIALS = [b"x%d" % i for i in range(256)]
    MSGS = [b"label-%d" % i for i in range(64)]
    VALUES = [(-1) ** i * (i % 17) for i in range(64)]

    def dense():
        return ext.dense_dots(MATERIALS, 24, 16, bytes(8 * 16), 1, 4)

    def r255():
        return ops.r255_mcfe_encrypt(MSGS, b"pyfe4ai-fork-test", VALUES, 12345, 67890)

    first = sys.argv[1]
    want = {"dense": dense(), "r255": r255()}
    for _ in range(3):
        assert (dense() if first == "dense" else r255()) == want[first]
    results = []
    for k in range(4):
        pid = os.fork()
        if pid == 0:
            try:
                for _ in range(3):
                    assert dense() == want["dense"]
                    assert r255() == want["r255"]
                os._exit(0)
            except BaseException as exc:
                print("child:", type(exc).__name__, exc, file=sys.stderr)
                os._exit(3)
        _, status = os.waitpid(pid, 0)
        results.append("signal %d" % os.WTERMSIG(status) if os.WIFSIGNALED(status)
                       else "exit %d" % os.WEXITSTATUS(status))
        assert (dense() if first == "dense" else r255()) == want[first]  # the parent goes on with 4 threads
    print(" ".join(results))
""")


@pytest.mark.parametrize("first", ["dense", "r255"])
def test_a_forked_child_runs_multi_threaded_kernels(first, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"), PYFE4AI_THREADS="4")
    run = subprocess.run([sys.executable, "-c", SCRIPT, first], capture_output=True, text=True, env=env,
                         cwd=tmp_path, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == ["exit", "0"] * 4, (run.stdout, run.stderr[-2000:])
