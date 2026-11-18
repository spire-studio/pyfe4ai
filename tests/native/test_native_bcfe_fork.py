"""The bcfe native kernels run with several threads in a forked child after
the parent ran them with several threads.

On macOS the standard library of Rust parks a thread on a dispatch semaphore
whose Mach port does not survive ``fork()``; a kernel that parked its caller
while waiting for its worker threads (``std::thread::scope``) made the next
such wait of the same thread in a forked child abort the process (SIGTRAP,
"BUG IN CLIENT OF LIBDISPATCH: Use-after-free of dispatch_semaphore_t"). The
``bcfe_*`` kernels join their workers instead
(``pyfe4ai_core::parallel_for_joined``), so the caller never parks. Each
case runs in a fresh interpreter (a crash must not take the test run with
it): the parent works with 4 threads, forks, and the child runs the slot
kernels with 4 threads and checks the decryption; the parent reports how the
child ended. Skipped without ``os.fork`` or the extension.
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
    from pyfe4ai import backend, circuits
    from pyfe4ai.primitives import garbling

    backend.set_threads(4)
    ops = backend.ops()
    pc = circuits.linear_threshold(n=4, bw=4, bx=4)
    c = pc.circuit
    enc, m = c.guard().encode(), c.m
    w = [(7 * j) % 3 % 2 for j in range(m)]
    x = [(5 * i) % 2 for i in range(c.n)]
    want = garbling.pack_bits(c.evaluate_plain(w, x), c.n_out)

    def slot_round(slot_id):
        wp, xp = garbling.pack_bits(w, m), garbling.pack_bits(x, c.n)
        s = ops.bcfe_new_slot(bytes(32), slot_id, wp, m=m)
        assert ops.bcfe_unselected_ok(bytes(32), slot_id, wp, s["k"], s["pks"], m=m)
        ct = ops.bcfe_encrypt_slot(enc, "hardwired", bytes(32), bytes(16), slot_id, xp, s["pks"])
        y = ops.bcfe_decrypt_slot(enc, "hardwired", bytes(32), bytes(16), slot_id, ct["R"], ct["label_cts"],
                                  ct["tables"], ct["decode"], ct["data_active"], s["scalars"], wp)
        assert y == want, (y, want)

    def other_kernel():
        # a multi-threaded kernel shared with the other schemes (dense-LWE label dots)
        backend.ext().dense_dots([b"x%d" % i for i in range(256)], 24, 16, bytes(8 * 16), 1, 4)

    parent = sys.argv[1]
    for _ in range(3):
        if parent == "bcfe":
            slot_round(1)
        else:
            other_kernel()
    results = []
    for k in range(4):
        pid = os.fork()
        if pid == 0:
            try:
                for i in range(3):
                    slot_round(10 + i)
                os._exit(0)
            except BaseException as exc:
                print("child:", type(exc).__name__, exc, file=sys.stderr)
                os._exit(3)
        _, status = os.waitpid(pid, 0)
        results.append("signal %d" % os.WTERMSIG(status) if os.WIFSIGNALED(status)
                       else "exit %d" % os.WEXITSTATUS(status))
        slot_round(20 + k)  # and the parent goes on with 4 threads
    print(" ".join(results))
""")


@pytest.mark.parametrize("parent", ["bcfe", "other"])
def test_a_forked_child_runs_the_slot_kernels_with_threads(parent, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYFE4AI_CACHE_DIR=str(tmp_path / "cache"), PYFE4AI_THREADS="4")
    run = subprocess.run([sys.executable, "-c", SCRIPT, parent], capture_output=True, text=True, env=env,
                         cwd=tmp_path, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == ["exit", "0"] * 4, (run.stdout, run.stderr[-2000:])
