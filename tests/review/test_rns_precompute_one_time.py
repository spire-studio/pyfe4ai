"""Review 2026-10, M1 and L3 for the RNS precomputation (exp/rns-modulus).

As tests/review/test_precompute_one_time.py of the review for the shared
precomputation: a native RNS precomputation made before ``os.fork()`` must
not be usable in the child (parent and child would encrypt different inputs
with the same ``r`` and noise, and ``ct0 - ct0' = Delta (x - x')``); and a
chunk index near ``2^64`` must be refused, not wrap around.
"""

import json
import os

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE, MCFERingLWEKeyGenerator
from pyfe4ai.utils import native_backend

NIDS = ["a", "b", "c"]
pytestmark = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with native_backend.backend("native"):
        yield


def _client(ring_n=64):
    kg = MCFERingLWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": ring_n,
                                  "bound_x": 100, "bound_y": 2, "ring_modulus": "rns"})
    kg.setup()
    pp = kg.get_public_parameters()
    assert "q_factors" in pp
    return pp, MCFERingLWE({"id": "a", "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys("a")}})


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_rns_precomputation_is_not_reused_after_fork():
    pp, enc = _client()
    pre = enc.precompute("round-7", [(4,)])
    assert pre.native is not None and type(pre.native[1]).__name__ == "RnsPreObj"
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: a second, different gradient under the same precomputation
        try:
            ct = enc.encrypt_lst_ndarray([np.array([-7, 2, 1, 1])], label="round-7", precomputed=pre)
            os.write(write, json.dumps(ct[0]["chunks"][0], default=str).encode())
        except Exception:  # refused
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


def test_rns_precomputation_chunk_index_does_not_wrap():
    _, enc = _client()
    pre = enc.precompute("L", [(64,)])
    nkey, cap = pre.native
    x = np.ones(64, dtype=np.int64)
    with pytest.raises(ValueError, match="does not cover"):
        native_backend.ext().encrypt_precomputed(nkey, cap, 2 ** 64 - 1, x, 1, False)
    assert native_backend.ext().precomputed_info(cap) == {"chunks": 1, "used": 0}
