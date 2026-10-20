"""The NTT-domain Ring-LWE paths (Python backend) give the same outputs as the v0.3.0 code.

``RingNTT`` caches ``NTT(a)``, ``NTT(pk)`` and ``NTT(sk_y)``, transforms ``r``
once and sums decryption products in the NTT domain. With the same Gaussian
samples (a seeded sampler patched into the scheme modules) the ciphertexts,
decryptions and threshold partial decryptions must equal those of the v0.3.0
implementations kept verbatim below -- also for 200+-bit moduli (the flooded
q of ``mcfe/ring_lwe_threshold``), which the native backend handles in
256-bit words.
"""

from __future__ import annotations

import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe_rlwe
from pyfe4ai.schemes.mcfe.labels import chunk_label
from pyfe4ai.schemes.mife import ring_lwe as mife_rlwe
from pyfe4ai.schemes.sife import ring_lwe as sife_rlwe
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.core.exceptions import FESchemeError
from pyfe4ai.primitives.lattice.lwe_utils import decode_scaled
from pyfe4ai.core.formats.binary import canonical_ring_ciphertext
from pyfe4ai.primitives.lattice.ring_lwe_utils import (
    mat_vec_mul,
    poly_add,
    poly_mul,
    poly_neg,
    transpose,
)
from pyfe4ai._validation import int_vector

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


@pytest.fixture(autouse=True)
def _python_backend():
    """These tests compare Python paths (tests/test_native_backend.py the native one)."""
    with native_backend.backend("python"):
        yield


class SeededGaussian:
    """``discrete_gaussian_matrix`` with a resettable seeded source."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def __call__(self, rows: int, cols: int, sigma: float, sampler=None) -> list:
        return [[gp.mpz(int(round(self.rng.gauss(0.0, sigma)))) for _ in range(cols)] for _ in range(rows)]


@pytest.fixture
def sampler(monkeypatch):
    s = SeededGaussian(2026)
    for mod in (mife_rlwe, sife_rlwe):
        monkeypatch.setattr(mod, "discrete_gaussian_matrix", s)
    return s


# --------------------------------------------------------------- v0.3.0 code

def ref_padded_rows(self, matrix_pt, extra_pad=None) -> list:
    q = gp.mpz(self.pp["q"])
    ring_n = self.pp["ring_n"]
    rows = []
    for i, (row, u_row) in enumerate(zip(matrix_pt, self.sk["u"])):
        padded = [gp.mpz(int(v)) for v in row] + [gp.mpz(0)] * (ring_n - len(row))
        extra = extra_pad[i] if extra_pad is not None else [0] * ring_n
        rows.append([(x + gp.mpz(u) + e) % q for x, u, e in zip(padded, u_row, extra)])
    return rows


def ref_encrypt_rows(self, w_rows: list, k: int, dgm) -> dict:
    q = gp.mpz(self.pp["q"])
    ring_n = self.pp["ring_n"]
    delta = q // gp.mpz(self.pp["p"])
    pk = [[gp.mpz(v) for v in row] for row in self.sk["pk"]]
    r = dgm(1, ring_n, self.pp["sigma2"])[0]
    noise = dgm(self.pp["eta"], ring_n, self.pp["sigma3"])
    ct0 = []
    for i, w_row in enumerate(w_rows):
        ct0_i = poly_add(poly_mul(pk[i], r, q), noise[i], q)
        ct0.append(poly_add(ct0_i, [delta * w for w in w_row], q))
    e = dgm(1, ring_n, self.pp["sigma2"])[0]
    ct1 = poly_add(poly_mul(self.A, r, q), e, q)
    return {
        "ct0": [[gp.digits(v) for v in row] for row in ct0],
        "ct1": [gp.digits(v) for v in ct1],
        "k": k,
    }


def ref_decrypt_with_pad_term(self, dct_ct, sk_y_map, z, fusion_weight, k) -> list:
    q = gp.mpz(self.pp["q"])
    delta = q // gp.mpz(self.pp["p"])
    total = [(-delta * gp.mpz(v)) % q for v in z]
    for nid, ct in dct_ct.items():
        y = int_vector(
            fusion_weight[nid], "fusion weight of {}".format(nid),
            length=self.pp["eta"], bound=int(self.pp["bound_y"]),
        )
        try:
            _, ct0, ct1 = canonical_ring_ciphertext(ct, int(q), self.pp["eta"], self.pp["ring_n"])
        except FESchemeError as exc:
            raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
        ct0 = [[gp.mpz(v) for v in row] for row in ct0]
        ct1 = [gp.mpz(v) for v in ct1]
        sk_y = [gp.mpz(v) for v in sk_y_map[nid]]
        y_vec = [gp.mpz(int(v)) for v in y]
        lhs = mat_vec_mul(transpose(ct0), y_vec, q)
        rhs = poly_mul(ct1, sk_y, q)
        total = poly_add(total, poly_add(lhs, poly_neg(rhs, q), q), q)
    bound = (
        self.pp["n"] * self.pp["eta"]
        * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
    )
    return [decode_scaled(v, delta, q, bound) for v in total[:k]]


def ref_mcfe_encrypt_lst_ndarray(self, lst_ndarray, label, dgm) -> list:
    from pyfe4ai._validation import scale_to_int
    import math

    ring_n = self.pp["ring_n"]
    out, chunk = [], 0
    for ary_src in lst_ndarray:
        flat = scale_to_int(ary_src, self.precision).ravel().tolist()
        n_layer = max(1, math.ceil(len(flat) / ring_n))
        cts = []
        for j in range(n_layer):
            part = flat[j * ring_n:(j + 1) * ring_n]
            self._check_plaintext([part])
            rows = ref_padded_rows(self, [part], extra_pad=self._pad(chunk_label(label, chunk + j)))
            cts.append(ref_encrypt_rows(self, rows, len(part), dgm))
        out.append({"shape": list(np.shape(ary_src)), "first_chunk": chunk, "chunks": cts})
        chunk += n_layer
    return out


# ------------------------------------------------------------------- setups

def _mife(eta: int, ring_n: int, bound_x: int = 3, bound_y: int = 3, n: int = 3, mcfe: bool = False,
          stat_param=None):
    """MIFE or MCFE Ring-LWE; with *stat_param*, the threshold MCFE (its MCFE
    layer on the flooded q: 200+ bits for stat_param >= 150)."""
    nids = ["c{}".format(i) for i in range(n)]
    mod = mcfe_rlwe if mcfe else mife_rlwe
    kg_cls = mod.MCFERingLWEKeyGenerator if mcfe else mod.MIFERingLWEKeyGenerator
    crypto_cls = mod.MCFERingLWE if mcfe else mod.MIFERingLWE
    cfg = {"n": n, "lst_nid": nids, "eta": eta, "ring_n": ring_n, "sec_param": 64,
           "bound_x": bound_x, "bound_y": bound_y}
    if stat_param is not None:
        kg_cls, crypto_cls = tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator, tmcfe_rlwe.ThresholdMCFERingLWE
        # the format and modulus of v0.3.0, whose code these tests compare with
        cfg.update(s=3, t=2, lst_sid=["s0", "s1", "s2"], stat_param=stat_param, ct_format="json",
                   ring_modulus="prime")
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: crypto_cls({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return kg, pp, clients, crypto_cls({"id": "agg", "precision": 0, "keys": {"pp": pp}})


@pytest.mark.parametrize("eta,ring_n", [(1, 16), (2, 32), (3, 64), (1, 256)])
def test_mife_encrypt_rows_identical(sampler, eta, ring_n):
    kg, pp, clients, _ = _mife(eta, ring_n)
    rng = random.Random(eta * ring_n)
    for nid, cl in clients.items():
        for k in (1, ring_n // 2, ring_n):
            x = [[rng.randint(-3, 3) for _ in range(k)] for _ in range(eta)]
            sampler.reset()
            new = cl._encrypt_rows(cl._padded_rows(x), k)
            sampler.reset()
            old = ref_encrypt_rows(cl, ref_padded_rows(cl, x), k, sampler)
            assert new == old


@pytest.mark.parametrize("eta,ring_n", [(1, 16), (2, 32), (3, 64)])
def test_mife_decrypt_identical_honest_and_tampered(sampler, eta, ring_n):
    kg, pp, clients, agg = _mife(eta, ring_n)
    rng = random.Random(7 + eta)
    q = int(pp["q"])
    weights = {nid: [rng.randint(-3, 3) for _ in range(eta)] for nid in clients}
    dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": weights})
    cts = {nid: cl.encrypt([[rng.randint(-3, 3) for _ in range(ring_n)] for _ in range(eta)])
           for nid, cl in clients.items()}
    new = agg._decrypt_with_pad_term(cts, dk["sk_y"], dk["z"], weights, ring_n)
    assert new == ref_decrypt_with_pad_term(agg, cts, dk["sk_y"], dk["z"], weights, ring_n)
    assert None not in new
    # random ciphertexts: most coefficients decode to None, the rest must agree too
    for trial in range(3):
        bad = {nid: {"ct0": [[str(rng.randrange(q)) for _ in range(ring_n)] for _ in range(eta)],
                     "ct1": [rng.randrange(-q, 2 * q) for _ in range(ring_n)], "k": ring_n}
               for nid in clients}
        sk_hats = {}
        new = agg._decrypt_with_pad_term(bad, dk["sk_y"], dk["z"], weights, ring_n, sk_hats)
        assert new == ref_decrypt_with_pad_term(agg, bad, dk["sk_y"], dk["z"], weights, ring_n)
        # the cached NTT(sk_y) gives the same result as a fresh transform
        assert agg._decrypt_with_pad_term(bad, dk["sk_y"], dk["z"], weights, ring_n, sk_hats) == new


@pytest.mark.parametrize("stat_param", [None, 150])
def test_mcfe_ndarray_identical(sampler, stat_param):
    kg, pp, clients, agg = _mife(1, 64, bound_x=100, bound_y=1, mcfe=True, stat_param=stat_param)
    if stat_param is not None:
        assert int(pp["q"]).bit_length() > 200
    for cl in list(clients.values()) + [agg]:
        cl.precision = 2
    rng = np.random.default_rng(3)
    grads = {nid: [rng.uniform(-1, 1, 150).round(2), rng.uniform(-1, 1, (3, 5)).round(2)] for nid in clients}
    enc = {}
    for nid, cl in clients.items():
        sampler.reset()  # the MCFE layer (no per-server tags of the threshold variant)
        enc[nid] = mcfe_rlwe.MCFERingLWE.encrypt_lst_ndarray(cl, grads[nid], label="r1")
        sampler.reset()
        # the v0.3.0 values; the layers also carry the label hash's code
        assert all(layer.pop("label_hash") == 2 for layer in enc[nid])
        assert enc[nid] == ref_mcfe_encrypt_lst_ndarray(cl, grads[nid], "r1", sampler)
        for layer in enc[nid]:
            layer["label_hash"] = 2
    weights = {nid: [1] for nid in clients}
    n_chunks = agg.key_chunks(next(iter(enc.values())))
    dk = mcfe_rlwe.MCFERingLWEKeyGenerator.get_decryption_keys(  # unshared, also for the threshold variant
        kg, "agg", credentials={"fusion_weight": weights, "label": "r1", "n_chunks": n_chunks})
    out = mcfe_rlwe.MCFERingLWE.decrypt_lst_ndarray_ct(agg, enc, dk=dk, fusion_weight=weights, label="r1")
    for l in range(2):
        assert np.allclose(out[l], sum(g[l] for g in grads.values()))
    # chunk by chunk against the v0.3.0 decryption
    for l, layer in enumerate(next(iter(enc.values()))):
        for j in range(len(layer["chunks"])):
            dct = {nid: enc[nid][l]["chunks"][j] for nid in enc}
            z = dk["z_chunks"][layer["first_chunk"] + j]
            k = dct["c0"]["k"]
            assert agg._decrypt_with_pad_term(dct, dk["sk_y"], z, weights, k) == \
                ref_decrypt_with_pad_term(agg, dct, dk["sk_y"], z, weights, k)


def test_mcfe_ndarray_bound_and_empty_errors(sampler):
    from pyfe4ai.core.exceptions import FEValidationError

    _, _, clients, _ = _mife(1, 16, bound_x=3, bound_y=1, mcfe=True)
    cl = next(iter(clients.values()))
    with pytest.raises(FEValidationError, match="exceeds the bound 3"):
        cl.encrypt_lst_ndarray([np.array([1, 2, 4])], label="x")
    with pytest.raises(FEValidationError, match="between 1 and ring_n"):
        cl.encrypt_lst_ndarray([np.array([])], label="y")


@pytest.mark.parametrize("flooding", ["uniform", "gaussian"])
@pytest.mark.parametrize("stat_param", [64, 150])
def test_threshold_share_decrypt_identical(sampler, monkeypatch, stat_param, flooding):
    nids, sids = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
    kg = tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator({
        "n": 3, "lst_nid": nids, "s": 3, "t": 2, "lst_sid": sids, "eta": 1, "ring_n": 32,
        "sec_param": 64, "bound_x": 4, "bound_y": 4, "stat_param": stat_param, "flooding": flooding,
        "ct_format": "json", "ring_modulus": "prime", "key_shares": "full"})  # as in v0.3.0
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tmcfe_rlwe.ThresholdMCFERingLWE({"id": nid, "precision": 0,
                                                    "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    weights = {nid: [1] for nid in nids}
    grads = {nid: np.arange(70) % 9 - 4 for nid in nids}
    enc = {nid: cl.encrypt_lst_ndarray([grads[nid]], label="r") for nid, cl in clients.items()}
    n_chunks = clients["c0"].key_chunks(enc["c0"])
    creds = {"fusion_weight": weights, "label": "r", "n_chunks": n_chunks}
    parts = {}
    for sid in sids[:2]:
        dk = kg.get_decryption_keys(sid, credentials=creds)
        server = tmcfe_rlwe.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}})
        new = server.compute_lst_ndarray_ct(enc, creds, dk, sids[:2])
        with monkeypatch.context() as m:  # the poly_mul path of v0.3.0
            m.setattr(tmcfe_rlwe.RingNTT, "get", classmethod(lambda cls, q, n: None))
            old = server.compute_lst_ndarray_ct(enc, creds, dk, sids[:2])
        assert new == old
        parts[sid] = new
    combiner = tmcfe_rlwe.ThresholdMCFERingLWE({"id": "comb", "precision": 0, "keys": {"pp": pp}})
    out = combiner.decrypt_lst_ndarray_ct(parts)
    assert np.allclose(out[0], sum(grads.values()))


def test_sife_encrypt_identical(sampler):
    kg = sife_rlwe.SIFERingLWEKeyGenerator({"eta": 2, "ring_n": 32, "sec_param": 64, "bound_x": 3, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    cl = sife_rlwe.SIFERingLWE({"id": "c", "keys": {"pp": pp, "sk": kg.get_private_keys("c")}})
    x = [[1, -2, 3], [0, 3, -3]]
    sampler.reset()
    new = cl.encrypt(x)
    sampler.reset()
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sife_rlwe.RingNTT, "get", classmethod(lambda cls, q, n: None))
        cl.__dict__.pop("_enc_key", None)
        old = cl.encrypt(x)
    assert new == old


@pytest.mark.parametrize("bits", [78, 212, 253])
def test_ring_ntt_matches_poly_mul(bits):
    q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), 64)
    ring = rlu.RingNTT.get(q, 64)
    rng = random.Random(1)
    a = [gp.mpz(rng.randrange(-int(q), int(q))) for _ in range(64)]
    b = [gp.mpz(rng.randrange(int(q))) for _ in range(64)]
    c = [gp.mpz(rng.randrange(int(q))) for _ in range(64)]
    acc = ring.mul_add(ring.mul(ring.forward(a), ring.forward(b)), ring.forward(c), ring.forward(b))
    assert ring.inverse(acc) == poly_add(poly_mul(a, b, q), poly_mul(c, b, q), q)
    assert rlu.RingNTT.get(gp.mpz(97), 64) is None  # 128 does not divide 96
