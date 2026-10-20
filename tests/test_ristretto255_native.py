"""The Rust ristretto255 (pyfe4ai._rust_native) gives the Python reference's results.

Group operations, the RFC 9380 hash and the dlog are compared encoding by
encoding; MCFE encryption over ristretto255 is deterministic, so the batched
native ndarray paths must return the Python ciphertexts and results exactly.
Skipped unless the Rust extension is built.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ddh_ristretto255 as mcfe_ec
from pyfe4ai.primitives.groups import ec_group
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives.groups import ristretto255 as R
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(native_backend.implementation() != "rust", reason="Rust extension not built"),
]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def both(fn):
    with native_backend.backend("python"):
        py = fn()
    with native_backend.backend("native"):
        nat = fn()
    return py, nat


def test_group_operations_identical():
    ext = native_backend.ext()
    rng = random.Random(1)
    assert ext.r255_base() == R.encode(R.BASE)
    for _ in range(20):
        a, b = rng.randrange(R.L), rng.randrange(R.L)
        pa, pb = R.mul_base(a), R.mul_base(b)
        ea, eb = R.encode(pa), R.encode(pb)
        assert ext.r255_mul_base(a.to_bytes(32, "little")) == ea
        assert ext.r255_mul(ea, b.to_bytes(32, "little")) == R.encode(R.mul(pa, b))
        assert ext.r255_add(ea, eb) == R.encode(R.add(pa, pb))
        assert ext.r255_sub(ea, eb) == R.encode(R.sub(pa, pb))
        assert ext.r255_neg(ea) == R.encode(R.neg(pa))
        assert ext.r255_lincomb([ea, eb], [b.to_bytes(32, "little"), a.to_bytes(32, "little")]) == \
            R.encode(R.add(R.mul(pa, b), R.mul(pb, a)))
        msg = rng.randbytes(rng.randrange(0, 80))
        assert ext.r255_hash(msg, b"DST") == R.encode(R.hash_to_group(msg, b"DST"))
        data = rng.randbytes(32)
        try:
            R.decode(data)
            valid = True
        except ValueError:
            valid = False
        assert ext.r255_check(data) == valid
    for x in (-1000, -3, 0, 5, 1000, 1001):
        assert both(lambda: ec_group.dlog(ec_group.mul_base(x), 1000)) == ((x, x) if abs(x) <= 1000 else (None, None))


def test_mcfe_ndarray_identical():
    nids = ["a", "b", "c"]
    kg = mcfe_ec.MCFERistretto255KeyGenerator({"n": 3, "lst_nid": nids})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {n: mcfe_ec.MCFERistretto255({"precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
               for n in nids}
    agg = mcfe_ec.MCFERistretto255({"precision": 2, "keys": {"pp": pp}})
    rng = np.random.default_rng(1)
    grads = {n: [rng.uniform(-1, 1, (3, 4)).round(2), rng.uniform(-1, 1, 5).round(2)] for n in nids}
    enc = {}
    for n, cl in clients.items():
        py, nat = both(lambda: cl.encrypt_lst_ndarray(grads[n], label="r"))
        assert all(np.array_equal(a, b) for a, b in zip(py, nat))
        enc[n] = nat
    weights = {n: [1] for n in nids}
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": weights})
    py, nat = both(lambda: agg.decrypt_lst_ndarray_ct(enc, dk, weights, "r"))
    for l in range(2):
        assert np.array_equal(py[l], nat[l])
        assert np.allclose(nat[l], sum(g[l] for g in grads.values()))
    enc["b"][1][2] = {"c": ["ff" * 32]}
    for name in ("python", "native"):
        with native_backend.backend(name), pytest.raises(FEValidationError, match="malformed ciphertext element of b"):
            agg.decrypt_lst_ndarray_ct(enc, dk, weights, "r")
    small = mcfe_ec.MCFERistretto255({"precision": 2, "dlog_bound": 10, "keys": {"pp": pp}})
    for name in ("python", "native"):
        with native_backend.backend(name), pytest.raises(FESchemeError, match="does not decrypt"):
            small.decrypt_lst_ndarray_ct({n: enc[n][:1] for n in nids}, dk, weights, "r")
