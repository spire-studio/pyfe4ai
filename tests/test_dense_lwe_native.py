"""Dense-LWE MCFE (ABMPSW20): the native label-hash dot products give the
same ciphertexts and decryptions as the Python code."""

from __future__ import annotations

import hashlib
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import lwe as mcfe_lwe
from pyfe4ai.schemes.mcfe import fullysec_lwe as mcfe_fslwe
from pyfe4ai.utils import crypto_utils, native_backend
from pyfe4ai.utils.exceptions import FESchemeError

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("bits", [40, 64, 77, 128, 187, 300])
def test_hash_to_zq_vector_unchanged(bits, monkeypatch):
    q = gp.next_prime(gp.mpz(2) ** (bits - 1))
    got = crypto_utils.hash_to_zq_vector("lab", q, 57, domain="d")
    monkeypatch.setattr(crypto_utils, "_MPZ_FROM_BYTES", None)  # the v0.3.0 reduction
    assert got == crypto_utils.hash_to_zq_vector("lab", q, 57, domain="d")
    nb = (bits + 135) // 8
    stream = hashlib.shake_256(crypto_utils.hash_to_zq_material("lab", "d")).digest(57 * nb)
    assert got == [gp.mpz(int.from_bytes(stream[k * nb:(k + 1) * nb], "big")) % q for k in range(57)]


def test_dot_mod_unchanged():
    rng = random.Random(2)
    q = gp.next_prime(gp.mpz(2) ** 186)
    left = [str(rng.randrange(-int(q), int(q))) for _ in range(100)] + [-5, gp.mpz(7)]
    right = [gp.mpz(rng.randrange(int(q))) for _ in range(102)]
    acc = gp.mpz(0)
    for a, b in zip(left, right):
        acc += gp.mpz(a) * b
    assert mcfe_lwe.dot_mod(left, right, q) == acc % q


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
@pytest.mark.parametrize("mod,cfg", [
    (mcfe_lwe, {"lwe_n": 16, "bound_x": 100, "bound_y": 2}),
    (mcfe_lwe, {"lwe_n": 64, "bound_x": 7, "bound_y": 7, "sec_param": 128, "stat_param": 128}),
    (mcfe_fslwe, {"lwe_n": 24, "bound_x": 100, "bound_y": 2}),
])
def test_native_dense_lwe_identical(mod, cfg):
    nids = ["c0", "c1", "c2"]
    kg_cls = getattr(mod, "MCFEFullySecLWEKeyGenerator", None) or mod.MCFELWEKeyGenerator
    cls = getattr(mod, "MCFEFullySecLWE", None) or mod.MCFELWE
    kg = kg_cls(dict(cfg, n=3, lst_nid=nids, eta=1))
    kg.setup()
    pp = kg.get_public_parameters()
    prec = 2 if cfg["bound_x"] == 100 else 0
    clients = {n: cls({"id": n, "precision": prec, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in nids}
    agg = cls({"id": "a", "precision": prec, "keys": {"pp": pp}})
    rng = np.random.default_rng(1)
    lim = cfg["bound_x"] / 10 ** prec
    grads = {n: [rng.uniform(-lim, lim, (4, 5)).round(prec), rng.uniform(-lim, lim, 3).round(prec)] for n in nids}
    enc = {}
    for n, cl in clients.items():
        with native_backend.backend("python"):
            py = cl.encrypt_lst_ndarray(grads[n], label="r")
        with native_backend.backend("native"):
            native_backend.set_threads(3)
            try:
                nat = cl.encrypt_lst_ndarray(grads[n], label="r")
            finally:
                native_backend.set_threads(1)
        assert all(np.array_equal(a, b) for a, b in zip(py, nat))
        enc[n] = nat
    weights = {n: [1] for n in nids}
    dk = kg.get_decryption_keys("a", credentials={"fusion_weight": weights, "label": "r"})
    outs = {}
    for name in ("python", "native"):
        with native_backend.backend(name):
            outs[name] = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="r")
    for l in range(2):
        assert np.array_equal(outs["python"][l], outs["native"][l])
        assert np.allclose(outs["native"][l], sum(g[l] for g in grads.values()))
    # a key for other weights decodes to garbage; the same garbage on both paths
    dk2 = kg.get_decryption_keys("a", credentials={"fusion_weight": {n: [2] for n in nids}, "label": "r"})
    res = {}
    for name in ("python", "native"):
        with native_backend.backend(name):
            try:
                res[name] = agg.decrypt_lst_ndarray_ct(enc, dk=dk2, fusion_weight=weights, label="r")
            except FESchemeError as exc:
                res[name] = str(exc)
    if isinstance(res["python"], str):
        assert res["python"] == res["native"]
    else:
        assert all(np.array_equal(a, b) for a, b in zip(res["python"], res["native"]))
