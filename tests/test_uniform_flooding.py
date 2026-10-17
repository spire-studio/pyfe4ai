"""Opt-in uniform flooding of the threshold Ring-LWE MCFE (``flooding="uniform"``).

* :func:`uniform_flooding_poly` follows its specification, keeps known
  answers, is uniform on ``[-2^bits, 2^bits)`` and depends on every seed byte.
* Parameters: ``2^tau > 2^stat_param B``, ``q`` holds ``s 2^tau`` and is
  smaller than the Gaussian's; separate parameter records; the default
  (Gaussian) public parameters are unchanged.
* Partial decryptions combine to the right sums on both backends, are
  identical across backends and repeatable, and differ per server and set.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tm
from pyfe4ai.utils import native_backend
from pyfe4ai.utils.exceptions import FEValidationError

pytestmark = pytest.mark.core
needs_native = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")

NIDS, SIDS = ["c0", "c1", "c2"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _spec(bits: int, n: int, seed: bytes) -> list:
    nb = (bits + 1 + 7) // 8
    stream = hashlib.shake_256(
        b"pyfe4ai/tmcfe-ring-lwe/uniform-flooding/v1" + len(seed).to_bytes(8, "big") + seed
    ).digest(nb * n)
    return [int.from_bytes(stream[i * nb:(i + 1) * nb], "big") % 2 ** (bits + 1) - 2 ** bits for i in range(n)]


@pytest.mark.parametrize("bits", [1, 7, 8, 63, 64, 200, 201])
def test_uniform_poly_follows_specification(bits):
    seed = bytes(range(40))
    got = tm.uniform_flooding_poly(bits, 300, seed)
    assert got == _spec(bits, 300, seed)
    assert all(-2 ** bits <= e < 2 ** bits for e in got)
    assert got != tm.uniform_flooding_poly(bits, 300, seed[:-1] + b"\xff")
    assert tm.uniform_flooding_poly(bits, 50, seed) == got[:50]


def test_uniform_poly_known_answer():
    vals = tm.uniform_flooding_poly(201, 64, b"kat-seed")
    assert hashlib.sha256(",".join(map(str, vals)).encode()).hexdigest() == "4852f436f1e11bb3145a6efed71a633b7c264e15cc57a1b44003083ba116a647"


def test_uniform_poly_is_uniform():
    """Chi-square over the 2^5 values of bits = 4 (p about 1e-6), and the
    mean and variance of a wide one."""
    vals = np.array(tm.uniform_flooding_poly(4, 32 * 2000, b"chi"))
    counts = np.bincount(vals + 16, minlength=32)
    chi2 = float(((counts - 2000) ** 2 / 2000).sum())
    assert counts.size == 32 and chi2 < 31 + 5 * (2 * 31) ** 0.5 + 20
    wide = np.array(tm.uniform_flooding_poly(60, 20000, b"wide"), dtype=float) / 2.0 ** 60
    assert abs(wide.mean()) < 0.02 and abs(wide.var() - 1 / 3) < 0.02


def _kg(flooding=None, ring_n=32, stat_param=40, bound_x=100):
    cfg = {"n": 3, "lst_nid": NIDS, "eta": 1, "ring_n": ring_n, "sec_param": 64, "bound_x": bound_x, "bound_y": 1,
           "s": 3, "t": 2, "lst_sid": SIDS, "stat_param": stat_param}
    if flooding is not None:
        cfg["flooding"] = flooding
    kg = tm.ThresholdMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    return kg, kg.get_public_parameters()


def test_parameters_and_records():
    kg_g, pp_g = _kg()
    assert "sigma_f" in pp_g and "flooding" not in pp_g and "flood_bits" not in pp_g
    kg_u, pp_u = _kg("uniform")
    assert pp_u["flooding"] == "uniform" and "sigma_f" not in pp_u
    tau, sigma_f = pp_u["flood_bits"], int(pp_g["sigma_f"])
    bound = sigma_f >> 40  # B
    assert 2 ** tau >= bound << 40 and 2 ** (tau - 1) <= bound << 40
    assert int(pp_u["q"]) < int(pp_g["q"])
    q, p = int(pp_u["q"]), int(pp_u["p"])
    assert q >= p * (2 * (bound + 3 * 2 ** tau) + 4)
    # each record is only reused by its own flooding; switching regenerates
    assert _kg("uniform")[1]["q"] == pp_u["q"] and _kg("uniform")[1]["A"] == pp_u["A"]
    assert _kg("gaussian")[1]["q"] == pp_g["q"]
    with pytest.raises(FEValidationError):
        _kg("laplace")


def _round(pp, kg, backend, ring_n=32, size=70, enrolled=("s0", "s2")):
    clients = {nid: tm.ThresholdMCFERingLWE({"id": nid, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    rng = np.random.default_rng(size)
    grads = {nid: rng.uniform(-1, 1, size).round(2) for nid in NIDS}
    with native_backend.backend(backend):
        cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label="r") for nid in NIDS}
        creds = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "r",
                 "n_chunks": tm.ThresholdMCFERingLWE.key_chunks(cts["c0"])}
        parts = {}
        for sid in enrolled:
            srv = tm.ThresholdMCFERingLWE({"id": sid, "precision": 2, "keys": {"pp": pp}})
            parts[sid] = srv.compute_lst_ndarray_ct(cts, creds, kg.get_decryption_keys(sid, credentials=creds),
                                                    list(enrolled))
        out = tm.ThresholdMCFERingLWE({"id": "x", "precision": 2, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(parts)
    assert np.allclose(out[0], sum(grads.values()))
    return cts, creds, parts


@pytest.mark.parametrize("backend", ["python", pytest.param("native", marks=needs_native)])
@pytest.mark.parametrize("stat_param,bound_x", [(40, 100), (128, 100), (64, 5000)])
def test_uniform_round_trip(backend, stat_param, bound_x):
    kg, pp = _kg("uniform", stat_param=stat_param, bound_x=bound_x)
    _round(pp, kg, backend)


@needs_native
def test_partials_identical_across_backends_and_repeatable():
    kg, pp = _kg("uniform", ring_n=64, stat_param=128)
    cts, creds, _ = _round(pp, kg, "python", ring_n=64, size=130)
    dk = kg.get_decryption_keys("s1", credentials=creds)
    srv = tm.ThresholdMCFERingLWE({"id": "s1", "precision": 2, "keys": {"pp": pp}})
    outs = []
    for name in ("python", "native", "native"):
        with native_backend.backend(name):
            outs.append(srv.compute_lst_ndarray_ct(cts, creds, dk, ["s0", "s1"]))
    assert outs[0] == outs[1] == outs[2]
    # another enrolled set: another lambda, other noise
    with native_backend.backend("native"):
        other = srv.compute_lst_ndarray_ct(cts, creds, dk, ["s1", "s2"])
    assert other[0]["parts"][0]["w"] != outs[0][0]["parts"][0]["w"]


def test_noise_is_the_seeded_uniform_polynomial():
    """w - lambda * (key part) equals uniform_flooding_poly of the request's
    seed: the Gaussian sampler is not involved."""
    kg, pp = _kg("uniform", ring_n=16, stat_param=40)
    calls = []
    real = tm.uniform_flooding_poly

    def spy(bits, n, seed):
        calls.append((bits, n))
        return real(bits, n, seed)

    import unittest.mock as mock

    with mock.patch.object(tm, "uniform_flooding_poly", spy), \
            mock.patch.object(tm, "_flooding_poly", side_effect=AssertionError("Gaussian flooding used")):
        _round(pp, kg, "python", ring_n=16, size=20)
    assert calls and all(c == (pp["flood_bits"], 16) for c in calls)


def test_malformed_flooding_parameters_refused():
    kg, pp = _kg("uniform", ring_n=16)
    cts, creds, _ = _round(pp, kg, "python", ring_n=16, size=10)
    dk = kg.get_decryption_keys("s0", credentials=creds)
    for bad in ({"flood_bits": 0}, {"flood_bits": "96"}, {"flooding": "laplace"}):
        srv = tm.ThresholdMCFERingLWE({"id": "s0", "precision": 2, "keys": {"pp": dict(pp, **bad)}})
        with pytest.raises(FEValidationError):
            srv.compute_lst_ndarray_ct(cts, creds, dk, ["s0", "s1"])
