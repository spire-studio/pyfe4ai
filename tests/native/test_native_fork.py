"""Native kernels in a forked child of a process that ran them with several threads.

On macOS the standard library of Rust parks a waiting thread on a libdispatch semaphore, and libdispatch refuses to
work in the child of a fork from a multi-threaded process: ``pyfe4ai_core::parallel_for`` parked its caller while its
workers finished, so the child of a process that had run any kernel with several threads was killed at its next
multi-threaded call (SIGTRAP, "BUG IN CLIENT OF LIBDISPATCH: Use-after-free of dispatch_semaphore_t", "crashed on
child side of fork pre-exec"). The native core now records the first process that runs several threads; in any other
process (a fork of it) the kernels run with one thread, whatever thread count is asked for later
(``pyfe4ai_core::thread_limit``).

Each case runs in a fresh interpreter, so that a crash cannot take the test run with it. The parent runs the kernels
with 4 threads and enough work that its caller waits for the workers; then it forks four times. Each child asks for
4 threads again (``set_threads(4)``), runs the kernels with 4 requested threads for three rounds, compares their
results with the parent's and checks that it is limited to one thread; one child also forks a grandchild that does
the same. The parent goes on with 4 threads after each fork and must exit normally with status 0. The kernels are
``dense_dots`` (dense-LWE label dots), ``r255_mcfe_encrypt`` (ristretto255 MCFE) and a BLS12-381 batch
deserialisation, none of them bcfe. A control case forks from a parent that never ran several threads: its children
keep their threads. Skipped without ``os.fork`` or the extension.
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
    from pyfe4ai.primitives.pairing import pairing_backend as pb
    from pyfe4ai.primitives.pairing.pairing_backend import G2, PairingGroup

    mode = sys.argv[1]                       # "threads": the parent runs 4 threads; "single": it never does
    backend.set_threads(4 if mode == "threads" else 1)
    ext, ops = backend.ext(), backend.ops()
    group = PairingGroup("BLS12-381")
    MATERIALS = [b"x%d" % i for i in range(4096)]
    MSGS = [b"label-%d" % i for i in range(512)]
    VALUES = [(-1) ** i * (i % 17) for i in range(512)]
    G2S = pb.serialize_many(group, [group.random(G2) for _ in range(64)])

    def run(threads):
        backend.set_threads(threads)
        return (ext.dense_dots(MATERIALS, 24, 16, bytes(8 * 16), 1, threads),
                ops.r255_mcfe_encrypt(MSGS, b"pyfe4ai-fork-test", VALUES, 12345, 67890),
                pb.serialize_many(group, pb.deserialize_many(group, G2S)))

    want = run(4 if mode == "threads" else 1)
    for _ in range(2):
        assert run(4 if mode == "threads" else 1) == want
    # a child of a parent that ran several threads is limited to one; a child of a parent that never did keeps
    # its threads (and, having run them, limits its own children); a grandchild is therefore always limited
    child_limit = 1 if mode == "threads" else 4

    def child(limit, grandchild):
        for _ in range(3):
            assert run(4) == want                       # 4 threads asked for, again after the fork
        assert ext.fork_thread_limit(4) == limit, ext.fork_thread_limit(4)
        if grandchild:
            pid = os.fork()
            if pid == 0:
                try:
                    os._exit(child(1, False))
                except BaseException as exc:
                    print("grandchild:", type(exc).__name__, exc, file=sys.stderr)
                    os._exit(3)
            _, st = os.waitpid(pid, 0)
            if not (os.WIFEXITED(st) and os.WEXITSTATUS(st) == 0):
                print("grandchild ended with", st, file=sys.stderr)
                return 4
        return 0

    results = []
    for k in range(4):
        pid = os.fork()
        if pid == 0:
            try:
                os._exit(child(child_limit, k == 0))
            except BaseException as exc:
                print("child:", type(exc).__name__, exc, file=sys.stderr)
                os._exit(3)
        _, status = os.waitpid(pid, 0)
        results.append("exit %d" % os.WEXITSTATUS(status) if os.WIFEXITED(status)
                       else "signal %d" % os.WTERMSIG(status))
        if mode == "threads":
            assert run(4) == want and ext.fork_thread_limit(4) == 4   # the parent keeps its threads
    print(" ".join(results))
""")


@pytest.mark.parametrize("mode", ["threads", "single"])
def test_forked_children_run_the_native_kernels(mode, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"), PYFE4AI_THREADS="4")
    run = subprocess.run([sys.executable, "-c", SCRIPT, mode], capture_output=True, text=True, env=env,
                         cwd=tmp_path, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == ["exit", "0"] * 4, (run.stdout, run.stderr[-2000:])
