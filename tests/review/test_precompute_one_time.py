"""One-time use of the Ring-LWE MCFE precomputation (October 2026 code review).

These tests state the property the offline / online API documents ("use it
once"), which did not hold before the fixes:

* a precomputation made before ``os.fork()`` was unused in both processes, so
  parent and child encrypted different gradients with the same ``r`` and
  noise, and ``ct0 - ct0' = Delta (x - x')`` gives the difference without any
  key;
* two threads using one native precomputation deadlocked (the second blocked
  on the precomputation's mutex while holding the GIL, the first needed the
  GIL after its kernel) instead of one of them being refused.
"""

import json
import os
import subprocess
import sys
import textwrap

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator

NIDS = ["a", "b", "c"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _client(ring_n=64):
    kg = MCFERingLWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": ring_n,
                                  "bound_x": 100, "bound_y": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    return pp, MCFERingLWE({"id": "a", "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys("a")}})


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_precomputation_is_not_reused_after_fork():
    pp, enc = _client()
    pre = enc.precompute("round-7", [(4,)])
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: a second, different gradient under the same precomputation
        try:
            ct = enc.encrypt_lst_ndarray([np.array([-7, 2, 1, 1])], label="round-7", precomputed=pre)
            os.write(write, json.dumps(ct[0]["chunks"][0], default=str).encode())
        except Exception:  # refused: what the fix should do
            os.write(write, b"null")
        os._exit(0)
    os.close(write)
    data = b""
    while chunk := os.read(read, 1 << 16):
        data += chunk
    os.waitpid(pid, 0)
    parent = enc.encrypt_lst_ndarray([np.array([5, -3, 0, 9])], label="round-7", precomputed=pre)[0]["chunks"][0]
    child = json.loads(data)
    if child is None:
        return
    q, p = gp.mpz(pp["q"]), gp.mpz(pp["p"])
    delta = q // p
    diff = []
    for a, b in zip(parent["ct0"][0][:4], child["ct0"][0][:4]):
        v = (gp.mpz(a) - gp.mpz(b)) % q
        diff.append(int(v if v <= q // 2 else v - q) // int(delta))
    assert parent["ct1"] != child["ct1"], "same r and noise in parent and child: x - x' = {}".format(diff)


_THREADS = textwrap.dedent(
    """
    import os, sys, threading
    import numpy as np
    os.chdir(sys.argv[1])
    os.environ["PYFE4AI_CACHE_DIR"] = os.path.join(sys.argv[1], "config")
    from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator
    from pyfe4ai import backend as native_backend
    nids = ["a", "b", "c"]
    kg = MCFERingLWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "lst_nid": nids, "ring_n": 1024,
                                  "bound_x": 100, "bound_y": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = MCFERingLWE({"id": "a", "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys("a")}})
    shape = (3000 * 1024,)
    pre = enc.precompute("L", [shape])
    if pre.native is None:
        print("python path", flush=True); os._exit(0)
    nkey, cap = pre.native
    x = np.ones(shape, dtype=np.int64)
    out, bar = [], threading.Barrier(2)
    def worker():
        bar.wait()
        try:
            native_backend.ext().encrypt_precomputed(nkey, cap, 0, x, 1, False)
            out.append("ok")
        except ValueError as e:
            out.append("refused")
    ts = [threading.Thread(target=worker) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    print(sorted(out), flush=True)
    os._exit(0)
    """
)


def test_concurrent_use_of_a_native_precomputation_does_not_deadlock(tmp_path):
    from pyfe4ai import backend as native_backend

    if native_backend.ext() is None:
        pytest.skip("native extension not built")
    # the native precomputation also under PYFE4AI_BACKEND=python (whose 3M-element
    # Python precomputation would take minutes with the FACCT sampler)
    env = dict(os.environ, PYFE4AI_BACKEND="native",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        res = subprocess.run([sys.executable, "-c", _THREADS, str(tmp_path)], capture_output=True, text=True,
                             timeout=120, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("two threads on one native precomputation deadlocked (GIL held while waiting for its mutex)")
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip() in ("['ok', 'refused']", "python path"), res.stdout
