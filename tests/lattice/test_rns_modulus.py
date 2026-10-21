"""The opt-in RNS modulus ``ring_modulus="rns"`` of the Ring-LWE MIFE / MCFE.

``q = q1 q2`` (two NTT primes): the derivation, the NTT root (CRT of the
primes' roots), the schemes on the Python reference path, the native code
(one 64-bit NTT per prime) against it bit for bit, the parameter cache and
the refusals. The prime path (default) is checked to be unchanged. See
docs/rns_modulus_review.md.
"""

from __future__ import annotations

import json
import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_decentralized as dmcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe_rlwe
from pyfe4ai.schemes.mife import ring_lwe as mife_rlwe
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu
from pyfe4ai.primitives import hashing as cu
from pyfe4ai.primitives import sampling as sampling_utils
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.base import cache_dir

pytestmark = pytest.mark.core
native_only = pytest.mark.skipif(not native_backend.available(), reason="native extension not built")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


@pytest.fixture(autouse=True)
def _defaults():
    """Compare from ``auto`` with the rounded sampler, whatever the environment selects."""
    select = getattr(sampling_utils, "gaussian_sampler", None)
    with native_backend.backend("auto"):
        if select is None:
            yield
        else:
            with select("rounded"):
                yield


class FixedBytes:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        self.pos += n
        assert self.pos <= len(self.data), "byte stream exhausted"
        return self.data[self.pos - n:self.pos]


def ByteStream(seed: int) -> FixedBytes:
    return FixedBytes(random.Random(seed).randbytes(1 << 22))


class StreamRandom(random.Random):
    """``SystemRandom.random`` on a :class:`ByteStream` (7 bytes per uniform)."""

    def __init__(self, stream) -> None:
        self.stream = stream
        super().__init__(0)

    def random(self) -> float:
        return (int.from_bytes(self.stream.take(7), "big") >> 3) * 2 ** -53


@pytest.fixture
def same_bytes(monkeypatch):
    """``run(fn, seed)``: *fn* on the Python and the native backend, both fed one byte stream."""

    def run(fn, seed=1):
        with native_backend.backend("python"):
            monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(seed)))
            py = fn()
        with native_backend.backend("native"):
            monkeypatch.setattr(native_backend, "_random_override", ByteStream(seed).take)
            nat = fn()
            monkeypatch.setattr(native_backend, "_random_override", None)
        return py, nat

    return run


def _setup(ring_n=16, bound_x=100, bound_y=1, eta=1, n=3, kind="mcfe", precision=0, **extra):
    nids = ["c{}".format(i) for i in range(n)]
    cfg = {"n": n, "lst_nid": nids, "eta": eta, "ring_n": ring_n, "sec_param": 128, "bound_x": bound_x,
           "bound_y": bound_y, "ring_modulus": "rns", **extra}
    kg_cls, cls = {
        "mife": (mife_rlwe.MIFERingLWEKeyGenerator, mife_rlwe.MIFERingLWE),
        "mcfe": (mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE),
        "dmcfe": (dmcfe_rlwe.DecentralizedMCFERingLWEKeyGenerator, dmcfe_rlwe.DecentralizedMCFERingLWE),
    }[kind]
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: cls({"id": nid, "precision": precision, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return kg, pp, clients, cls({"id": "agg", "precision": precision, "keys": {"pp": pp}})


# ------------------------------------------------------------ derivation


@pytest.mark.parametrize("ring_n,bound_x,clients", [(2048, 100, 3), (4096, 100, 3), (4096, 2, 3), (8192, 100, 10),
                                                    (16, 100, 3), (64, 3, 2)])
def test_rns_modulus_derivation(ring_n, bound_x, clients):
    args = (128, clients, bound_x, 1, ring_n)
    prime = rlu.bmms21_selective_parameters(*args, security_bound_x=3 * bound_x)
    rns = rlu.bmms21_selective_parameters(*args, security_bound_x=3 * bound_x, ring_modulus="rns")
    q, (q1, q2) = rns["q"], rns["q_factors"]
    assert q == q1 * q2 and q1 < q2 and gp.is_prime(q1) and gp.is_prime(q2)
    assert (q1 - 1) % (2 * ring_n) == 0 and (q2 - 1) % (2 * ring_n) == 0
    assert q2.bit_length() <= rlu.RNS_MAX_FACTOR_BITS
    assert gp.gcd(q // rns["p"], q) == 1
    # the same bound as the prime: q >= q_prime's lower bound, and no more bits
    lower = rlu.prev_ntt_prime(prime["q"], ring_n)  # the prime is the first NTT prime above the bound
    assert q > lower and q.bit_length() == prime["q"].bit_length()
    assert {k: v for k, v in rns.items() if k not in ("q", "q_factors")} == \
        {k: v for k, v in prime.items() if k != "q"}
    assert "q_factors" not in prime
    assert rlu.bmms21_selective_parameters(*args, security_bound_x=3 * bound_x, ring_modulus="rns") == rns


def test_prime_roots_unchanged():
    """The stricter root check (root^n = -1) picks the same root for every prime."""
    for bits in (20, 40, 61, 62, 63, 77, 80, 125, 212):
        for n in (16, 1024, 4096, 8192):
            q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), n)
            e = (q - 1) // (2 * n)
            old = next(r for r in (gp.powmod(g, e, q) for g in range(2, 1000))
                       if r != 1 and gp.powmod(r, 2 * n, q) == 1 and gp.powmod(r, n, q) != 1)
            assert rlu._find_primitive_root(q, 2 * n) == old


def test_composite_root_and_products():
    q, fs = rlu.rns_ntt_modulus(gp.mpz(2) ** 79 + 12345, 32, 602)
    assert rlu._find_primitive_root(q, 64) is None  # no root without the factors
    rlu.register_rns_modulus(q, fs)
    root = rlu._get_ntt_root(q, 32)
    assert gp.powmod(root, 32, q) == q - 1 and all(gp.powmod(root, 32, f) == f - 1 for f in fs)
    rng = random.Random(4)
    a = [gp.mpz(rng.randrange(q)) for _ in range(32)]
    b = [gp.mpz(rng.randrange(q)) for _ in range(32)]
    assert rlu.poly_mul(a, b, q) == rlu.poly_mul_negacyclic(a, b, q)
    ring = rlu.RingNTT.get(q, 32)
    assert ring.inverse(ring.forward(a)) == a


def test_register_rns_modulus_checks():
    q, (q1, q2) = rlu.rns_ntt_modulus(gp.mpz(2) ** 60, 16, 14)
    for bad in ([q1], [q1, q1], [q1, q2 + 32], [q1 * q2, 1], [3, q // 3] if q % 3 == 0 else [q1, q2, 1], ["x", 1]):
        with pytest.raises(FEValidationError):
            rlu.register_rns_modulus(q, bad)
    assert rlu.register_rns_modulus(q, [str(q1), str(q2)]) == (q1, q2)
    with pytest.raises(FEValidationError):
        rlu.rns_ntt_modulus(gp.mpz(2) ** 130, 16, 14)  # factors beyond 62 bits
    with pytest.raises(FEValidationError):
        rlu.check_ring_modulus("crt")


# ------------------------------------------------------- schemes, Python


@pytest.mark.parametrize("kind", ["mife", "mcfe"])
def test_python_schemes_decrypt(kind):
    with native_backend.backend("python"):
        kg, pp, clients, agg = _setup(ring_n=32, bound_x=7, bound_y=3, eta=2, kind=kind)
        assert int(pp["q"]) == int(pp["q_factors"][0]) * int(pp["q_factors"][1])
        rng = random.Random(1)
        w = {nid: [rng.randint(-3, 3) for _ in range(2)] for nid in clients}
        xs = {nid: [[rng.randint(-7, 7) for _ in range(20)] for _ in range(2)] for nid in clients}
        want = [sum(y * x[i] for nid in clients for y, x in zip(w[nid], xs[nid])) for i in range(20)]
        if kind == "mife":
            dk = kg.get_decryption_keys("a", credentials={"fusion_weight": w})
            got = agg.decrypt({nid: cl.encrypt(xs[nid]) for nid, cl in clients.items()}, dk, w)
        else:
            dk = kg.get_decryption_keys("a", credentials={"fusion_weight": w, "label": "L"})
            got = agg.decrypt({nid: cl.encrypt(xs[nid], "L") for nid, cl in clients.items()}, dk, w, "L")
        assert got == want


@pytest.mark.parametrize("backend", ["python", pytest.param("native", marks=native_only)])
def test_decentralized_rns(backend):
    with native_backend.backend(backend):
        kg, pp, clients, agg = _setup(ring_n=32, bound_x=100, kind="dmcfe", precision=2)
        rng = np.random.default_rng(2)
        grads = {nid: [rng.uniform(-1, 1, 70).round(2)] for nid in clients}
        enc = {nid: cl.encrypt_lst_ndarray(grads[nid], label="r") for nid, cl in clients.items()}
        weights = {nid: [1] for nid in clients}
        n_chunks = agg.key_chunks(enc["c0"])
        shares = {nid: cl.derive_function_decryption_key_share(weights, "r", n_chunks) for nid, cl in clients.items()}
        dk = agg.combine_function_decryption_key_share(shares)
        out = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="r")
        assert np.allclose(out[0], sum(g[0] for g in grads.values()))


def test_unknown_values_refused():
    # the threshold MCFE takes "rns" too (its flooded q has more factors: tests/threshold/test_threshold_rns.py)
    cfg = {"n": 3, "eta": 1, "ring_n": 16, "bound_x": 10, "bound_y": 1, "s": 3, "t": 2, "ring_modulus": "rns"}
    assert len(tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator(cfg).q_factors) >= 2
    with pytest.raises(FEValidationError, match="ring_modulus"):
        tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator(dict(cfg, ring_modulus="composite"))
    with pytest.raises(FEValidationError, match="ring_modulus"):
        mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "eta": 1, "ring_n": 16, "ring_modulus": "composite"})
    # the SIFE schemes refuse it rather than ignore it (adaptive SIFE: Ring-LHL for prime q)
    from pyfe4ai.schemes.sife import fullysec_ring_lwe as sife_fs
    from pyfe4ai.schemes.sife import ring_lwe as sife_rlwe

    sife = {"eta": 1, "ring_n": 16, "sec_param": 64, "bound_x": 3, "bound_y": 3, "ring_modulus": "rns"}
    with pytest.raises(FEValidationError, match="ring_modulus"):
        sife_rlwe.SIFERingLWEKeyGenerator(sife)
    with pytest.raises(FEValidationError, match="ring_modulus"):
        sife_fs.SIFEFullySecRingLWEKeyGenerator(sife)
    assert sife_rlwe.SIFERingLWEKeyGenerator(dict(sife, ring_modulus="prime")).q.bit_length() > 0


def test_parameter_cache():
    base = {"n": 3, "eta": 1, "ring_n": 16, "sec_param": 128, "bound_x": 100, "bound_y": 1}
    folder = os.path.join(cache_dir(), "authority", mcfe_rlwe.MCFERingLWEKeyGenerator._scheme_type, "param.json")
    rns_file = os.path.join(os.path.dirname(folder), "param-rns.json")
    prime = mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base))
    rec_prime = json.load(open(folder))
    assert "ring_modulus" not in rec_prime and "q_factors" not in rec_prime  # the default's records are unchanged
    prime.setup()
    assert "q_factors" not in prime.get_public_parameters()
    rns = mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base, ring_modulus="rns"))  # a record of its own, beside the prime's
    assert json.load(open(folder)) == rec_prime
    rec = json.load(open(rns_file))
    assert rec["ring_modulus"] == "rns" and int(rec["q"]) == int(rec["q_factors"][0]) * int(rec["q_factors"][1])
    assert rec["q"] != rec_prime["q"] and rns.q == int(rec["q"])
    again = mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base, ring_modulus="rns"))
    assert again.a_seed == rns.a_seed  # loaded
    # alternating the configurations loads both records instead of regenerating them
    assert mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base)).a_seed == prime.a_seed
    assert mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base, ring_modulus="rns")).a_seed == rns.a_seed
    # a record with other factors (or without them) is regenerated
    for tamper in (lambda r: r.update(q_factors=list(reversed(r["q_factors"]))), lambda r: r.pop("q_factors")):
        r = dict(rec)
        tamper(r)
        json.dump(r, open(rns_file, "w"))
        assert mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base, ring_modulus="rns")).a_seed != rec["A_seed"]
    # a prime record in the RNS file (or the other way round) is not taken
    json.dump(rec_prime, open(rns_file, "w"))
    assert mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base, ring_modulus="rns")).a_seed != rec_prime["A_seed"]
    json.dump(rec, open(folder, "w"))
    assert mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base)).a_seed != rec["A_seed"]
    assert mcfe_rlwe.MCFERingLWEKeyGenerator(dict(base)).q == prime.q


def test_standard_profile_auto():
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "eta": 1, "bound_x": 100, "bound_y": 1, "ring_n": "auto",
                                            "security_profile": "standard", "ring_modulus": "rns"})
    prime = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "eta": 1, "bound_x": 100, "bound_y": 1, "ring_n": "auto",
                                               "security_profile": "standard"})
    assert kg.ring_n == prime.ring_n == 4096 and kg.q.bit_length() == prime.q.bit_length() == 80
    assert len(kg.q_factors) == 2 and max(int(f).bit_length() for f in kg.q_factors) <= 40


# ------------------------------------------------- native against Python


@native_only
def test_native_context():
    kg, pp, _, _ = _setup()
    q = int(pp["q"])
    ctx = native_impl.ring_context(q, 16)
    info = native_backend.ext().ctx_info(ctx.cap)
    assert info["rns"] == [int(f) for f in pp["q_factors"]] and ctx.word_bytes == 16
    vals = [0, 1, q - 1, q, -1, 2 ** 100, "17", gp.mpz(q + 5)]
    assert ctx.unpack(ctx.pack(vals)) == [int(v) % q for v in vals]
    assert ctx.digits(ctx.pack([5, q - 2])) == ["5", str(q - 2)]
    assert len(ctx.ntt_mont(list(range(16)))) == 2 * 16 * 8
    # the NTT, the product and the label hash on the RNS context: the Python values
    ext = native_backend.ext()
    rng = random.Random(5)
    a, b = [rng.randrange(q) for _ in range(16)], [rng.randrange(q) for _ in range(16)]
    with native_backend.backend("python"):
        root = rlu._get_ntt_root(gp.mpz(q), 16)
        want_ntt = [int(v) for v in rlu.ntt_forward([gp.mpz(v) for v in a], gp.mpz(q), root)]
        want_mul = [int(v) for v in rlu.poly_mul([gp.mpz(v) for v in a], [gp.mpz(v) for v in b], gp.mpz(q))]
    fwd = ctx.unpack(ext.ntt(ctx.cap, ctx.pack(a), False))
    assert fwd == want_ntt
    assert ctx.unpack(ext.ntt(ctx.cap, ctx.pack(fwd), True)) == a
    assert ctx.unpack(ext.poly_mul(ctx.cap, ctx.pack(a), ctx.pack(b))) == want_mul
    for mode, code in cu.LABEL_HASH_CODES.items():
        material = cu.hash_to_zq_material("v", domain="d", mode=mode)
        want = [int(v) for v in cu.hash_to_zq_vector("v", gp.mpz(q), 40, domain="d", mode=mode)]
        assert ctx.unpack(ext.hash_to_zq(ctx.cap, material, 40, hash_mode=code)) == want
    with native_backend.backend("python"):
        assert native_impl.ring_context(q, 16) is None


CONFIGS = [  # (ring_n, bound_x, bound_y, eta)
    (16, 100, 1, 1),
    (32, 7, 3, 2),
    (64, 2, 2, 1),
    (16, 2 ** 20, 2 ** 10, 1),  # q of about 120 bits: factors of 60 bits
]


@native_only
@pytest.mark.parametrize("cfg", CONFIGS, ids=str)
def test_encrypt_identical(cfg, same_bytes):
    ring_n, bound_x, bound_y, eta = cfg
    kg, pp, clients, _ = _setup(ring_n, bound_x, bound_y, eta)
    assert native_impl.ring_context(pp["q"], ring_n) is not None
    rng = random.Random(5)
    for i, cl in enumerate(clients.values()):
        k = [1, ring_n // 2, ring_n][i]
        x = [[rng.randint(-bound_x, bound_x) for _ in range(k)] for _ in range(eta)]
        py, nat = same_bytes(lambda: cl.encrypt(x, "lab"), seed=i)
        assert py == nat
        for fmt in ("json", "binary"):
            cl.ct_format = fmt
            py, nat = same_bytes(lambda: cl._encrypt_rows(cl._padded_rows(x), k), seed=10 + i)
            assert py == nat
        cl.ct_format = "json"


@native_only
@pytest.mark.parametrize("ring_n,size", [(16, 1), (16, 37), (64, 300)])
def test_ndarray_encrypt_and_precompute_identical(ring_n, size, same_bytes):
    _, _, clients, _ = _setup(ring_n, precision=2)
    cl = clients["c0"]
    rng = np.random.default_rng(size)
    layers = [rng.uniform(-1, 1, size).round(2), rng.uniform(-1, 1, (2, 3)).round(2)]
    shapes = [np.shape(a) for a in layers]
    for fmt in ("json", "binary"):
        cl.ct_format = fmt
        py, nat = same_bytes(lambda: cl.encrypt_lst_ndarray(layers, label="r"))
        assert py == nat
        with native_backend.backend("native"):
            native_backend.set_threads(4)
            try:
                _, threaded = same_bytes(lambda: cl.encrypt_lst_ndarray(layers, label="r"))
            finally:
                native_backend.set_threads(1)
        assert threaded == py
        # the offline / online split gives the ciphertexts of a direct encryption with the same samples
        py_pre, nat_pre = same_bytes(lambda: cl.encrypt_lst_ndarray(layers, label="r",
                                                                     precomputed=cl.precompute("r", shapes)))
        assert py_pre == nat_pre == py
    with native_backend.backend("native"):
        pre = cl.precompute("r", shapes)
        chunks = sum(-(-int(np.prod(s)) // ring_n) for s in shapes)
        assert native_backend.ext().precomputed_info(pre.native[1]) == {"chunks": chunks, "used": 0}
        cl.encrypt_lst_ndarray(layers, label="r", precomputed=pre)
        assert native_backend.ext().precomputed_info(pre.native[1]) == {"chunks": chunks, "used": chunks}


@native_only
@pytest.mark.parametrize("cfg", CONFIGS, ids=str)
def test_keys_and_decryption_identical(cfg):
    ring_n, bound_x, bound_y, eta = cfg
    kg, pp, clients, agg = _setup(ring_n, bound_x, bound_y, eta)
    rng = random.Random(3)
    q = int(pp["q"])
    weights = {nid: [rng.randint(-bound_y, bound_y) for _ in range(eta)] for nid in clients}
    weights["c0"] = [1] * eta
    creds = {"fusion_weight": weights, "label": "L", "n_chunks": 2}
    with native_backend.backend("python"):
        dk_py = kg.get_decryption_keys("a", credentials=creds)
    with native_backend.backend("native"):
        dk = kg.get_decryption_keys("a", credentials=creds)
    assert dk == dk_py
    xs = {nid: [[rng.randint(-bound_x, bound_x) for _ in range(ring_n)] for _ in range(eta)] for nid in clients}
    want = [sum(y * x[i] for nid in clients for y, x in zip(weights[nid], xs[nid])) for i in range(ring_n)]

    def both(cts, k=ring_n):
        out = []
        for name in ("python", "native"):
            with native_backend.backend(name):
                out.append(agg._decrypt_with_pad_term(cts, dk["sk_y"], dk["z"], weights, k))
        assert out[0] == out[1]
        return out[1]

    for fmt in ("json", "binary"):
        for cl in clients.values():
            cl.ct_format = fmt
        cts = {nid: cl.encrypt(xs[nid], "L") for nid, cl in clients.items()}
        assert both(cts) == want
        assert both(cts, ring_n // 2) == want[:ring_n // 2]
    for _ in range(2):  # random ciphertexts mostly do not decode
        bad = {nid: {"ct0": [[str(rng.randrange(q)) for _ in range(ring_n)] for _ in range(eta)],
                     "ct1": [rng.randrange(-q, 2 * q) for _ in range(ring_n)], "k": ring_n} for nid in clients}
        both(bad)
    for cl in clients.values():
        cl.ct_format = "json"
    cts = {nid: cl.encrypt(xs[nid], "L") for nid, cl in clients.items()}
    delta, bound = q // int(pp["p"]), 3 * eta * bound_x * bound_y
    row0 = list(cts["c0"]["ct0"][0])
    row0[1] = str((int(row0[1]) + delta * (bound + 1 - want[1])) % q)
    got = both(dict(cts, c0=dict(cts["c0"], ct0=[row0] + cts["c0"]["ct0"][1:])))
    assert got[1] is None and got.count(None) == 1


@native_only
def test_e2e_standard_profile_native_threads_and_binary():
    from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = ["c0", "c1", "c2"]
    with native_backend.backend("native"):
        w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": nids, "bound_x": 100, "bound_y": 1,
                                                 "security_profile": "standard", "ring_modulus": "rns"},
                            precision=2, crypto_options={"ct_format": "binary"})
        rng = np.random.default_rng(0)
        for r, threads in enumerate((1, 3)):
            native_backend.set_threads(threads)
            try:
                g = {n: rng.uniform(-1, 1, 9000).round(2) for n in nids}
                enc = {n: encrypt_gradient(g[n], w, nid=n, label="r%d" % r) for n in nids}
                out = aggregate_gradients(enc, w, {n: 1 for n in nids}, label="r%d" % r)
            finally:
                native_backend.set_threads(1)
            assert np.allclose(out, sum(g.values()))


# ------------------------------------- the default sampler and label hashes


class _StreamBits:
    """``getrandbits`` over a fixed byte string, as the FACCT reference draws them."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def getrandbits(self, k: int) -> int:
        if k <= 0:
            return 0
        n = (k + 7) // 8
        block = self.data[self.pos:self.pos + n]
        assert len(block) == n, "byte stream exhausted"
        self.pos += n
        return int.from_bytes(block, "big") >> (8 * n - k)


@pytest.fixture
def same_bytes_facct(monkeypatch):
    """``run(fn, seed)``: *fn* on the Python reference and the native RNS kernels,
    FACCT fed the same bytes; the native calls must draw FACCT noise."""

    def run(fn, seed=1):
        data = random.Random(seed).randbytes(1 << 22)
        with native_backend.backend("python"):
            monkeypatch.setattr(sampling_utils, "_OS_RANDOM", _StreamBits(data))
            py = fn()
        used = []
        original = native_impl.noise_kwargs
        monkeypatch.setattr(native_impl, "noise_kwargs",
                            lambda *s, **kw: used.append(original(*s, **kw)) or used[-1])
        with native_backend.backend("native"):
            monkeypatch.setattr(native_backend, "_random_override", lambda n: data)
            nat = fn()
            monkeypatch.setattr(native_backend, "_random_override", None)
        monkeypatch.setattr(native_impl, "noise_kwargs", original)
        assert used and all("facct" in kw for kw in used)
        return py, nat

    return run


@native_only
@pytest.mark.parametrize("mode", cu.LABEL_HASHES)
def test_rns_mcfe_facct_and_label_hash_identical(mode, same_bytes_facct):
    """MCFE over an RNS modulus with the exact sampler (the default of the
    family) and each label hash: single-vector and ndarray encryption, the
    offline / online split and the key terms give the Python values, and the
    aggregate decrypts."""
    kg, pp, clients, agg = _setup(16, precision=0, gaussian_sampler="facct", label_hash=mode)
    assert pp["gaussian_sampler"] == "facct" and pp["label_hash"] == mode
    assert native_impl.ring_context(pp["q"], 16) is not None
    cl = clients["c0"]
    x = [[3, -1, 0, 7]]
    py, nat = same_bytes_facct(lambda: cl.encrypt(x, "lab"), seed=2)
    assert py == nat
    layer = np.random.default_rng(3).integers(-100, 101, 37)
    py, nat = same_bytes_facct(lambda: cl.encrypt_lst_ndarray([layer], label="r"), seed=3)
    assert py == nat
    py_pre, nat_pre = same_bytes_facct(
        lambda: cl.encrypt_lst_ndarray([layer], label="r", precomputed=cl.precompute("r", [layer.shape])), seed=3)
    assert py_pre == nat_pre == py
    creds = {"fusion_weight": {nid: [1] for nid in clients}, "label": "r", "n_chunks": cl.key_chunks(nat)}
    keys = []
    for name in ("python", "native"):
        with native_backend.backend(name):
            keys.append(kg.get_decryption_keys("agg", credentials=creds))
    assert keys[0] == keys[1]
    layers = {nid: np.random.default_rng(i).integers(-100, 101, 37) for i, nid in enumerate(clients)}
    cts = {nid: c.encrypt_lst_ndarray([layers[nid]], label="r") for nid, c in clients.items()}
    out = agg.decrypt_lst_ndarray_ct(cts, dk=keys[1], fusion_weight=creds["fusion_weight"], label="r")
    assert np.array_equal(out[0], sum(layers.values()))


@native_only
def test_rns_mife_facct_identical(same_bytes_facct):
    _, pp, clients, _ = _setup(32, kind="mife", eta=2, bound_x=7, bound_y=3, gaussian_sampler="facct")
    rng = random.Random(9)
    x = [[rng.randint(-7, 7) for _ in range(32)] for _ in range(2)]
    py, nat = same_bytes_facct(lambda: clients["c0"].encrypt(x), seed=9)
    assert py == nat



def _negacyclic(a, b, q):
    n = len(a)
    out = [0] * n
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                k = i + j
                if k < n:
                    out[k] += ai * bj
                else:
                    out[k - n] -= ai * bj
    return [v % q for v in out]


@native_only
@pytest.mark.parametrize("mode", cu.LABEL_HASHES)
def test_rns_drbg_path_facct_tensor_is_the_reference(mode):
    """The production path (rnd=None, the call's DRBG): chunk j of an RNS
    tensor encryption reads stream j from offset 0 and samples r, e_0, e as the
    Python FACCT reference does, whatever the thread count; the offline /
    online split gives the direct encryption (review of the integration)."""
    rng = random.Random(7)
    n = 64
    q, fs = rlu.rns_ntt_modulus(gp.mpz(1) << 100, n, 7)
    q, fs = int(q), tuple(int(f) for f in fs)
    rlu.register_rns_modulus(q, fs)
    ctx = native_impl.ring_context(q, n)
    ext = native_backend.ext()
    sig2 = 3.0e6
    sig3 = 2 ** 0.5 * sig2
    tables = native_impl.noise_kwargs(sig2, sig3, sampler="facct")["facct"]
    delta = q // 7
    a, pk, u = ([rng.randrange(q) for _ in range(n)] for _ in range(3))
    key = ext.enc_key(ctx.cap, a, [pk], [u], delta, hash_mode=cu.LABEL_HASH_CODES[mode])
    size = 3 * n + 5
    x = np.array([rng.randint(-3, 3) for _ in range(size)], dtype=np.int64)
    chunks = -(-size // n)
    mats = [cu.hash_to_zq_material("lab|%d" % j, domain="t", mode=mode) for j in range(chunks)]
    seed = bytes(rng.randrange(256) for _ in range(32))
    for threads in (1, 3):
        out = ext.encrypt_tensor(key, x, mats, sig2, sig3, None, threads, False, seed=seed, facct=tables)
        for j in range(chunks):
            h = [int(v) for v in cu.hash_to_zq_vector("lab|%d" % j, gp.mpz(q), n, domain="t", mode=mode)]
            xs = [int(v) for v in x[j * n:(j + 1) * n]] + [0] * max(0, (j + 1) * n - size)
            w = [(xs[i] + u[i] + h[i]) % q for i in range(n)]
            with native_backend.backend("python"):
                bits = _StreamBits(ext.drbg_bytes(seed, j, 1 << 16))
                r = [int(v) for v in sampling_utils.discrete_gaussian_facct(sig2, n, rng=bits)]
                e0 = [int(v) for v in sampling_utils.discrete_gaussian_facct(sig3, n, rng=bits)]
                e1 = [int(v) for v in sampling_utils.discrete_gaussian_facct(sig2, n, rng=bits)]
            pr, ar = _negacyclic(pk, r, q), _negacyclic(a, r, q)
            ct0 = [(pr[i] + e0[i] + delta * w[i]) % q for i in range(n)]
            ct1 = [(ar[i] + e1[i]) % q for i in range(n)]
            assert [int(v) for v in out[j][0][0]] == ct0 and [int(v) for v in out[j][1]] == ct1, (j, threads)
    pre = ext.precompute_tensor(key, mats, sig2, sig3, None, 2, seed=seed, facct=tables)
    direct = ext.encrypt_tensor(key, x, mats, sig2, sig3, None, 2, False, seed=seed, facct=tables)
    assert ext.encrypt_precomputed(key, pre, 0, x, 2) == direct
    with pytest.raises(ValueError):
        ext.encrypt_precomputed(key, pre, 0, x, 1)  # one-time


@native_only
@pytest.mark.parametrize("modulus", ["rns", "prime"])
def test_native_error_paths_of_the_wiping_functions(modulus):
    """lincomb, pad_terms and enc_key copy secret rows (pads, u', key pieces)
    and wipe them on every path: their refusals still raise as before."""
    n = 16
    if modulus == "rns":
        q, fs = rlu.rns_ntt_modulus(gp.mpz(1) << 60, n, 7)
        rlu.register_rns_modulus(q, fs)
    else:
        q = rlu.next_ntt_prime(gp.mpz(1) << 60, n)
    q = int(q)
    ctx = native_impl.ring_context(q, n)
    ext, cap = native_backend.ext(), ctx.cap
    row = ctx.pack(list(range(n)))
    assert ext.lincomb(cap, [row, row], [1, 2]) == [str(3 * i % q) for i in range(n)]
    with pytest.raises(ValueError, match="one coefficient per row"):
        ext.lincomb(cap, [row, row], [1])
    with pytest.raises(TypeError):
        ext.lincomb(cap, [row, 3], [1, 1])
    with pytest.raises(ValueError, match="one material per slot"):
        ext.pad_terms(cap, [0] * n, [[b"a"]], [[1], [2]], 1, 1)
    with pytest.raises(TypeError, match="materials must be bytes"):
        ext.pad_terms(cap, [0] * n, [[b"a", 5]], [[1], [2]], 1, 1)
    a = list(range(n))
    with pytest.raises(ValueError, match="one pad row per pk row"):
        ext.enc_key(cap, a, [a], [a, a], q // 7)
    with pytest.raises(ValueError):
        ext.enc_key(cap, a, [a], [a], q)  # Delta = q = 0 mod q
    # a malformed later row (read_rows wipes the rows parsed before it)
    with pytest.raises(TypeError):
        ext.enc_key(cap, a, [a, a], [a, [object()] * n], q // 7)
    key = ext.enc_key(cap, a, [a, a], [a, a], q // 7)
    with pytest.raises(TypeError):
        ext.encrypt(key, [a, [object()] * n], 3.0, 4.0, None)
    tables = native_impl.noise_kwargs(3.0, 4.0, sampler="facct")["facct"]
    with pytest.raises(ValueError, match="not enough random bytes"):
        ext.encrypt(key, [a, a], 3.0, 4.0, b"\0" * 16, facct=tables)
    with pytest.raises(ValueError, match="not enough random bytes"):
        ext.encrypt(key, [a, a], 3.0, 4.0, b"\0" * 16)
    assert ext.encrypt(key, [a, a], 3.0, 4.0, None, facct=tables) is not None
    if modulus == "prime":  # Shamir dealing (prime contexts only)
        with pytest.raises(ValueError, match="not enough random bytes"):
            ext.shamir_share(cap, list(range(8)), 3, [1, 2, 3], b"\0" * 9)
        assert len(ext.shamir_share(cap, list(range(8)), 3, [1, 2, 3], None)) == 3


@native_only
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_rns_precomputation_info_refuses_another_process():
    _, pp, clients, _ = _setup(16, precision=2)
    pre = clients["c0"].precompute("r", [(8,)])
    cap = pre.native[1]
    assert native_backend.ext().precomputed_info(cap)["used"] == 0
    pid = os.fork()
    if pid == 0:
        try:
            native_backend.ext().precomputed_info(cap)
            os._exit(3)
        except ValueError:
            os._exit(0)
        except BaseException:
            os._exit(4)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
