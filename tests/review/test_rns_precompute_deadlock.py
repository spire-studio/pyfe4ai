"""Review 2026-10 (exp/rns-modulus 256b6a9): the RNS precomputation has the
mutex-across-``py.detach`` pattern of main (rust/native/src/rns.rs
encrypt_precomputed / precomputed_info): two threads encrypting with one
precomputation through the public API deadlock the process instead of one of
them being refused. Fails on the reviewed commit.
"""

import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT = textwrap.dedent(
    """
    import os, sys, threading
    import numpy as np
    os.chdir(sys.argv[1])
    from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
    nids = ["c0", "c1", "c2"]
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": nids, "eta": 1, "ring_n": 1024, "sec_param": 128,
                                            "bound_x": 100, "bound_y": 1, "ring_modulus": "rns"})
    kg.setup()
    pp = kg.get_public_parameters()
    cl = mcfe_rlwe.MCFERingLWE({"id": "c0", "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys("c0")}})
    size = 1024 * 3000
    pre = cl.precompute("L", [(size,)])
    if pre.native is None:
        print("python path", flush=True); os._exit(0)
    out, bar = [], threading.Barrier(2)
    def go():
        bar.wait()
        try:
            cl.encrypt_lst_ndarray([np.zeros(size)], label="L", precomputed=pre); out.append("ok")
        except Exception:
            out.append("refused")
    ts = [threading.Thread(target=go) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    print(sorted(out), flush=True)
    os._exit(0)
    """
)


def test_concurrent_use_of_an_rns_precomputation_does_not_deadlock(tmp_path):
    # the native precomputation also under PYFE4AI_BACKEND=python (whose 3M-element
    # Python precomputation takes minutes with the FACCT sampler), as the prime twin
    env = dict(os.environ, PYFE4AI_BACKEND="native",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        res = subprocess.run([sys.executable, "-c", _SCRIPT, str(tmp_path)], capture_output=True, text=True,
                             timeout=150, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("two threads on one RNS precomputation deadlocked")
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip() in ("['ok', 'refused']", "python path"), res.stdout
