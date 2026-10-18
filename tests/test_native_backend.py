"""The native Ring-LWE backend computes exactly what the Python code computes.

Kernels (NTT, SHAKE-256, hash to ``Z_q``, the rounded Gaussian on the same
bytes) and the scheme paths (MIFE / MCFE encryption, decryption including
values that do not decode, key derivation) are compared with the Python
implementation; random bytes are fed to both from one deterministic stream.
Moduli up to 253 bits (64-, 128- and 256-bit words); the 200+-bit scheme
paths use the flooded moduli of ``mcfe/ring_lwe_threshold``. Skipped when the
extension is not built.
"""

from __future__ import annotations

import hashlib
import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe_rlwe
from pyfe4ai.schemes.mife import ring_lwe as mife_rlwe
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils import sampling_utils
from pyfe4ai.utils.crypto_utils import hash_to_zq_material, hash_to_zq_vector
from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not native_backend.available(), reason="native extension not built"),
]
N = native_backend.ext()


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _compare_backends():
    """Start from ``auto`` even when ``PYFE4AI_BACKEND=python`` runs the rest of
    the suite on the Python backend: these tests compare the two."""
    with native_backend.backend("auto"):
        yield


@pytest.fixture(autouse=True)
def _rounded_sampler():
    """The native sampler reproduces the default rounded Gaussian: compare with
    it even when another sampler is selected (PYFE4AI_GAUSSIAN_SAMPLER)."""
    select = getattr(sampling_utils, "gaussian_sampler", None)
    if select is None:  # versions without a sampler choice
        yield
        return
    with select("rounded"):
        yield


class FixedBytes:
    """The bytes of *data*, handed out in order."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        self.pos += n
        if self.pos > len(self.data):
            raise AssertionError("byte stream exhausted")
        return self.data[self.pos - n:self.pos]


def ByteStream(seed: int) -> FixedBytes:
    """A deterministic stream (sliced, so 7 + 7 bytes equal 14)."""
    return FixedBytes(random.Random(seed).randbytes(1 << 22))


class StreamRandom(random.Random):
    """``SystemRandom.random`` on a :class:`ByteStream` (7 bytes per uniform)."""

    def __init__(self, stream: ByteStream) -> None:
        self.stream = stream
        super().__init__(0)

    def random(self) -> float:
        return (int.from_bytes(self.stream.take(7), "big") >> 3) * 2 ** -53


@pytest.fixture
def same_bytes(monkeypatch):
    """``run(fn, seed)``: call *fn* with the Python and the native sampler fed
    the same byte stream; returns ``(python_result, native_result)``."""

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


def _ctx(bits: int, n: int):
    q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), n)
    return q, native_backend.RingContext(q, n, rlu._get_ntt_root(q, n))


def _word_bits(q) -> int:
    return 64 if q < 2 ** 63 else 128 if q < 2 ** 125 else 256


# ---------------------------------------------------------------- kernels

@pytest.mark.parametrize("bits,n", [(20, 2), (40, 16), (61, 256), (62, 1024), (63, 64), (77, 1024), (124, 512),
                                    (125, 16), (126, 16), (128, 32), (129, 64), (160, 256), (212, 1024),
                                    (212, 8192), (253, 32)])
def test_ntt_and_poly_mul_identical(bits, n):
    q, ctx = _ctx(bits, n)
    assert q.bit_length() == bits
    root = rlu._get_ntt_root(q, n)
    rng = random.Random(bits * n)
    a = [gp.mpz(rng.randrange(-2 * int(q), 2 * int(q))) for _ in range(n)]
    b = [gp.mpz(rng.randrange(int(q))) for _ in range(n)]
    pa, pb = ctx.pack(a), ctx.pack(b)
    assert ctx.word_bits == _word_bits(q)
    assert ctx.unpack(N.ntt(ctx.cap, pa)) == [int(v) for v in rlu.ntt_forward([v % q for v in a], q, root)]
    assert ctx.unpack(N.ntt(ctx.cap, pb, True)) == [int(v) for v in rlu.ntt_inverse(b, q, root)]
    assert ctx.unpack(N.poly_mul(ctx.cap, pa, pb)) == [int(v) for v in rlu.poly_mul(a, b, q)]
    assert ctx.digits(pa) == [gp.digits(v % q) for v in a]
    assert ctx.unpack(ctx.pack([gp.digits(v) for v in a])) == [int(v % q) for v in a]


@pytest.mark.parametrize("bits", [77, 129, 212, 253])
def test_pack_reads_like_int(bits):
    q, ctx = _ctx(bits, 16)
    big = [2 ** 90, -(2 ** 100), 2 ** 255 + 3, -(2 ** 300), int(q) + 5, -int(q) - 5, 2 ** 256 - 1, 2 ** 256]
    vals = ["12", "-5", " 7 ", "+3", "1_000", "007", "-0", gp.mpz(-9), np.int64(-4)] + big \
        + [str(v) for v in big] + [gp.mpz(v) for v in big[:3]]
    ints = [12, -5, 7, 3, 1000, 7, 0, -9, -4] + big + big + big[:3]
    vals, ints = vals + [0] * (32 - len(vals)), ints + [0] * (32 - len(ints))
    assert ctx.unpack(ctx.pack(vals)) == [v % int(q) for v in ints]
    assert ctx.digits(ctx.pack(vals)) == [str(v % int(q)) for v in ints]
    with pytest.raises(ValueError):
        ctx.pack(["12a"] + ["0"] * 15)


@pytest.mark.parametrize("length", [0, 1, 135, 136, 137, 271, 272, 5000])
def test_shake256_matches_hashlib(length):
    msg = random.Random(length).randbytes(length)
    for out in (1, 135, 136, 137, 4097):
        assert N.shake256(msg, out) == hashlib.shake_256(msg).digest(out)


@pytest.mark.parametrize("bits", [40, 61, 62, 63, 64, 77, 124, 126, 129, 192, 212, 253])
def test_hash_to_zq_identical(bits):
    q, ctx = _ctx(bits, 64)
    for v, dom in (("x", ""), ("u'|label", "pyfe4ai/mcfe/ring_lwe/H(u',label)"), ("é" * 50, "d")):
        got = ctx.unpack(N.hash_to_zq(ctx.cap, hash_to_zq_material(v, dom), 300))
        assert got == [int(x) for x in hash_to_zq_vector(v, q, 300, domain=dom)]


@pytest.mark.parametrize("sigma", [0.5, 3.2, 17.25, 4.97e9, 2.0 ** 40])
@pytest.mark.parametrize("count", [1, 2, 7, 1024])
def test_gaussian_identical_to_random_gauss(sigma, count):
    data = random.Random(count).randbytes(native_backend.gauss_bytes(count))
    ref = StreamRandom(FixedBytes(data))
    expected = [int(round(ref.gauss(0.0, sigma))) for _ in range(count)]
    assert N.gauss(data, count, sigma) == expected
    assert len(N.gauss(None, count, sigma)) == count  # OS bytes


def test_rejects_unsupported_moduli():
    q = rlu.next_ntt_prime(gp.mpz(2) ** 253, 16)  # 254 bits
    with pytest.raises(ValueError):
        native_backend.RingContext(q, 16, rlu._get_ntt_root(q, 16))
    assert native_backend.ring_context(q, 16) is None
    with pytest.raises(ValueError):  # 3 is not a primitive 2n-th root of unity
        native_backend.RingContext(rlu.next_ntt_prime(gp.mpz(2) ** 40, 16), 16, 3)


# ------------------------------------------------------------ scheme paths

CONFIGS = [  # (eta, ring_n, sec_param, bound_x, bound_y[, stat_param]): 64-, 128- and 256-bit words
    (1, 16, 64, 2, 2),
    (2, 32, 64, 3, 3),
    (3, 16, 64, 3, 3),
    (1, 64, 64, 50, 1),        # q of 62 bits: u64 words, lazy butterflies
    (1, 64, 64, 70, 1),        # q of 63 bits: u64 words, reduced butterflies
    (1, 64, 128, 100, 1),
    (2, 64, 128, 100, 3),
    (1, 16, 64, 50, 4, 64),    # tMCFE layer, q of 129 bits
    (1, 32, 64, 100, 1, 150),  # tMCFE layer, q of 216 bits
    (2, 16, 64, 3, 3, 170),    # tMCFE layer, q of 225 bits
]


def _setup(eta, ring_n, sec_param, bound_x, bound_y, stat_param=None, mcfe=True, n=3, precision=0):
    """MIFE / MCFE Ring-LWE; with *stat_param*, the threshold MCFE, whose MCFE
    layer (encryption, keys, decryption) runs on its flooded q."""
    nids = ["c{}".format(i) for i in range(n)]
    cfg = {"n": n, "lst_nid": nids, "eta": eta, "ring_n": ring_n, "sec_param": sec_param,
           "bound_x": bound_x, "bound_y": bound_y}
    if stat_param is not None:
        kg_cls, cls = tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator, tmcfe_rlwe.ThresholdMCFERingLWE
        cfg.update(s=3, t=2, lst_sid=["s0", "s1", "s2"], stat_param=stat_param)
    elif mcfe:
        kg_cls, cls = mcfe_rlwe.MCFERingLWEKeyGenerator, mcfe_rlwe.MCFERingLWE
    else:
        kg_cls, cls = mife_rlwe.MIFERingLWEKeyGenerator, mife_rlwe.MIFERingLWE
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: cls({"id": nid, "precision": precision, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return kg, pp, clients, cls({"id": "agg", "precision": precision, "keys": {"pp": pp}})


def _mcfe_keys(kg, sid, credentials):
    """The MCFE functional key (of a threshold key generator: unshared)."""
    return mcfe_rlwe.MCFERingLWEKeyGenerator.get_decryption_keys(kg, sid, credentials=credentials)


@pytest.mark.parametrize("cfg", CONFIGS, ids=str)
def test_encrypt_identical(cfg, same_bytes):
    eta, ring_n, _, bound_x, _ = cfg[:5]
    kg, pp, clients, _ = _setup(*cfg)
    assert native_backend.ring_context(pp["q"], ring_n).word_bits == _word_bits(int(pp["q"]))
    rng = random.Random(5)
    for i, cl in enumerate(clients.values()):
        k = [1, ring_n // 2, ring_n][i]
        x = [[rng.randint(-bound_x, bound_x) for _ in range(k)] for _ in range(eta)]
        py, nat = same_bytes(lambda: cl.encrypt(x, "lab"), seed=i)
        assert py == nat
        py, nat = same_bytes(lambda: cl._encrypt_rows(cl._padded_rows(x), k), seed=10 + i)
        assert py == nat


@pytest.mark.parametrize("ring_n,size,stat_param", [(16, 1, None), (16, 16, None), (16, 37, None), (64, 300, None),
                                                    (16, 37, 150), (64, 300, 170)])
def test_mcfe_ndarray_encrypt_identical(ring_n, size, stat_param, same_bytes):
    kg, pp, clients, _ = _setup(1, ring_n, 128, 100, 1, stat_param, precision=2)
    cl = clients["c0"]
    rng = np.random.default_rng(size)
    layers = [rng.uniform(-1, 1, size).round(2), rng.uniform(-1, 1, (2, 3)).round(2)]
    py, nat = same_bytes(lambda: cl.encrypt_lst_ndarray(layers, label="r"))
    assert py == nat
    with native_backend.backend("native"):
        native_backend.set_threads(4)
        try:
            _, threaded = same_bytes(lambda: cl.encrypt_lst_ndarray(layers, label="r"))
        finally:
            native_backend.set_threads(1)
    assert threaded == py


@pytest.mark.parametrize("cfg", CONFIGS, ids=str)
def test_keys_identical(cfg):
    eta = cfg[0]
    kg, pp, clients, _ = _setup(*cfg)
    rng = random.Random(9)
    weights = {nid: [rng.randint(-cfg[4], cfg[4]) for _ in range(eta)] for nid in clients}
    creds = {"fusion_weight": weights, "label": "L", "n_chunks": 3}
    with native_backend.backend("python"):
        py = _mcfe_keys(kg, "s", creds)
    with native_backend.backend("native"):
        nat = _mcfe_keys(kg, "s", creds)
    assert py == nat
    bad = dict(weights, c1=[cfg[4] + 1] * eta)
    for name in ("python", "native"):
        with native_backend.backend(name), pytest.raises(FEValidationError):
            _mcfe_keys(kg, "s", dict(creds, fusion_weight=bad))


def _decrypt_both(agg, *args):
    with native_backend.backend("python"):
        py = agg._decrypt_with_pad_term(*args)
    with native_backend.backend("native"):
        nat = agg._decrypt_with_pad_term(*args)
    return py, nat


@pytest.mark.parametrize("cfg", CONFIGS, ids=str)
def test_decrypt_identical_honest_and_random(cfg):
    eta, ring_n, _, bound_x, bound_y = cfg[:5]
    kg, pp, clients, agg = _setup(*cfg)
    rng = random.Random(3)
    q = int(pp["q"])
    weights = {nid: [rng.randint(-bound_y, bound_y) for _ in range(eta)] for nid in clients}
    weights["c0"] = [1] * eta
    dk = _mcfe_keys(kg, "agg", {"fusion_weight": weights, "label": "L"})
    xs = {nid: [[rng.randint(-bound_x, bound_x) for _ in range(ring_n)] for _ in range(eta)] for nid in clients}
    cts = {nid: cl.encrypt(xs[nid], "L") for nid, cl in clients.items()}
    py, nat = _decrypt_both(agg, cts, dk["sk_y"], dk["z"], weights, ring_n)
    assert py == nat and None not in nat
    py, nat = _decrypt_both(agg, cts, dk["sk_y"], dk["z"], weights, ring_n // 2)
    assert py == nat
    for trial in range(3):  # most values of random ciphertexts do not decode
        bad = {nid: {"ct0": [[str(rng.randrange(q)) for _ in range(ring_n)] for _ in range(eta)],
                     "ct1": [rng.randrange(-q, 2 * q) for _ in range(ring_n)], "k": ring_n}
               for nid in clients}
        py, nat = _decrypt_both(agg, bad, dk["sk_y"], dk["z"], weights, ring_n)
        assert py == nat
    # move results 1 and 3 to bound + 1 (weight 1 on row 0 of slot c0): they do not decode
    delta, bound = q // int(pp["p"]), 3 * eta * bound_x * bound_y
    row0 = list(cts["c0"]["ct0"][0])
    for i in (1, 3):
        mu = sum(y * x[i] for nid in clients for y, x in zip(weights[nid], xs[nid]))
        row0[i] = str((int(row0[i]) + delta * (bound + 1 - mu)) % q)
    shifted = dict(cts, c0=dict(cts["c0"], ct0=[row0] + cts["c0"]["ct0"][1:]))
    py, nat = _decrypt_both(agg, shifted, dk["sk_y"], dk["z"], weights, ring_n)
    assert py == nat and nat[1] is None and nat[3] is None and nat.count(None) == 2


def test_decrypt_value_formats_and_errors():
    kg, pp, clients, agg = _setup(1, 16, 64, 2, 2)
    weights = {nid: [1] for nid in clients}
    dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": weights, "label": "L"})
    cts = {nid: cl.encrypt([[1] * 16], "L") for nid, cl in clients.items()}
    q = int(pp["q"])

    def variant(f):
        out = {nid: {"ct0": [list(ct["ct0"][0])], "ct1": list(ct["ct1"]), "k": ct["k"]} for nid, ct in cts.items()}
        f(out["c1"])
        return out

    ok_variants = [
        lambda c: c["ct1"].__setitem__(0, " {} ".format(c["ct1"][0])),     # int() strips blanks
        lambda c: c["ct1"].__setitem__(1, "+" + c["ct1"][1]),
        lambda c: c["ct1"].__setitem__(2, int(c["ct1"][2]) - q),           # ints of any range
        lambda c: c["ct0"][0].__setitem__(3, gp.mpz(c["ct0"][0][3]) + q),
        lambda c: c.__setitem__("ct1", tuple(c["ct1"])),
        lambda c: c.__setitem__("ct0", (c["ct0"][0],)),
        lambda c: c["ct1"].__setitem__(4, "٣"),                            # non-ASCII digit
    ]
    for f in ok_variants:
        py, nat = _decrypt_both(agg, variant(f), dk["sk_y"], dk["z"], weights, 16)
        assert py == nat
    bad_variants = [
        lambda c: c["ct1"].__setitem__(0, "12a"),
        lambda c: c["ct1"].__setitem__(0, 1.5),
        lambda c: c["ct1"].__setitem__(0, None),
        lambda c: c["ct1"].pop(),
        lambda c: c["ct0"].append(c["ct0"][0]),
        lambda c: c.pop("ct1"),
        lambda c: c.__setitem__("k", 17),
        lambda c: c.__setitem__("k", True),
        lambda c: c.__setitem__("ct0", "x"),
    ]
    for f in bad_variants:
        for name in ("python", "native"):
            with native_backend.backend(name), pytest.raises(FESchemeError, match="malformed ciphertext of c1"):
                agg._decrypt_with_pad_term(variant(f), dk["sk_y"], dk["z"], weights, 16)
    for name in ("python", "native"):
        with native_backend.backend(name), pytest.raises(ValueError):
            agg._decrypt_with_pad_term(cts, dk["sk_y"], ["x"] + dk["z"][1:], weights, 16)


def test_mcfe_ndarray_aggregation_and_failures():
    kg, pp, clients, agg = _setup(1, 16, 128, 100, 1, precision=2)
    rng = np.random.default_rng(1)
    grads = {nid: [rng.uniform(-1, 1, 40).round(2), rng.uniform(-1, 1, (3, 2)).round(2)] for nid in clients}
    with native_backend.backend("native"):
        enc = {nid: cl.encrypt_lst_ndarray(grads[nid], label="r") for nid, cl in clients.items()}
    weights = {nid: [1] for nid in clients}
    n_chunks = agg.key_chunks(enc["c0"])
    dk = kg.get_decryption_keys("a", credentials={"fusion_weight": weights, "label": "r", "n_chunks": n_chunks})
    outs = {}
    for name in ("python", "native"):
        with native_backend.backend(name):
            outs[name] = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="r")
    for l in range(2):
        assert outs["python"][l].dtype == outs["native"][l].dtype == np.float64
        assert np.array_equal(outs["python"][l], outs["native"][l])
        assert np.allclose(outs["native"][l], sum(g[l] for g in grads.values()))
    # element 21 (chunk 1, position 5) moved to bound + 1 does not decode
    q, delta = int(pp["q"]), int(pp["q"]) // int(pp["p"])
    mu = int(round(sum(g[0][21] for g in grads.values()) * 100))
    row = enc["c2"][0]["chunks"][1]["ct0"][0]
    row[5] = str((int(row[5]) + delta * (3 * 100 + 1 - mu)) % q)
    for name in ("python", "native"):
        with native_backend.backend(name), pytest.raises(FESchemeError, match=r"element \(21,\) of layer 0"):
            agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="r")


def test_e2e_standard_profile_native_and_threads():
    from pyfe4ai.utils.ml_adapter import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = ["c0", "c1", "c2"]
    with native_backend.backend("native"):
        w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": nids, "bound_x": 100,
                                                 "bound_y": 1, "security_profile": "standard"}, precision=2)
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


def test_backend_selection(monkeypatch):
    with pytest.raises(ValueError):
        native_backend.set_backend("fast")
    with native_backend.backend("python"):
        assert native_backend.active_backend() == "python"
        assert native_backend.ring_context(rlu.next_ntt_prime(gp.mpz(2) ** 40, 16), 16) is None
    assert native_backend.describe()["available"]
    # the exact sampler is native too (tests/test_native_facct.py); another one stays in Python
    assert native_backend.sampling_ok(3.2, sampler="facct")
    assert native_backend.noise_kwargs(3.2, 4.0, sampler="facct")["facct"]
    assert not native_backend.sampling_ok(3.2, sampler="other")
    assert native_backend.noise_kwargs(3.2, 4.0, sampler="rounded") == {}
    assert native_backend.sampling_ok(3.2, 2.0 ** 40, sampler="rounded")
    assert not native_backend.sampling_ok(2.0 ** 41, sampler="rounded")
    with pytest.raises(TypeError):  # the scheme's sampler is required, not the process default
        native_backend.noise_kwargs(3.2, 4.0)
    with pytest.raises(TypeError):
        native_backend.sampling_ok(3.2)
    with pytest.raises(ValueError):
        native_backend.set_threads(0)


# ------------------------------------------------------------ threshold MCFE

TCONFIGS = [  # (ring_n, bound_x, bound_y, stat_param): q of 116 (128-bit words), 129 and 216 bits
    (16, 4, 4, 64),
    (16, 50, 4, 64),
    (32, 100, 3, 150),
]


@pytest.fixture
def native_calls(monkeypatch):
    """Counts the extension's calls by name (to see that a native path ran)."""
    import collections

    real, counts = native_backend._ext, collections.Counter()

    class Spy:
        def __getattr__(self, name):
            fn = getattr(real, name)

            def call(*args, **kwargs):
                counts[name] += 1
                return fn(*args, **kwargs)

            return call if callable(fn) else fn

    monkeypatch.setattr(native_backend, "_ext", Spy())
    return counts


def _threshold(ring_n, bound_x, bound_y, stat_param, eta=1, flooding="uniform"):
    nids, sids = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
    kg = tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator({
        "n": 3, "lst_nid": nids, "s": 3, "t": 2, "lst_sid": sids, "eta": eta, "ring_n": ring_n,
        "sec_param": 64, "bound_x": bound_x, "bound_y": bound_y, "stat_param": stat_param, "flooding": flooding})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: tmcfe_rlwe.ThresholdMCFERingLWE({"id": nid, "precision": 0,
                                                    "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return kg, pp, clients, sids


def _server(pp, sid):
    return tmcfe_rlwe.ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}})


def _both(fn):
    with native_backend.backend("python"):
        py = fn()
    with native_backend.backend("native"):
        nat = fn()
    return py, nat


@pytest.mark.parametrize("flooding", ["uniform", "gaussian"])
@pytest.mark.parametrize("cfg", TCONFIGS, ids=str)
def test_threshold_partials_and_combination_identical(cfg, flooding, native_calls):
    ring_n, bound_x, bound_y, _ = cfg
    kg, pp, clients, sids = _threshold(*cfg, flooding=flooding)
    rng = np.random.default_rng(4)
    grads = {nid: [rng.integers(-bound_x, bound_x + 1, 2 * ring_n + 3), rng.integers(-bound_x, bound_x + 1, (2, 3))]
             for nid in clients}
    weights = {nid: [int(rng.integers(-bound_y, bound_y + 1))] for nid in clients}
    enc = {nid: cl.encrypt_lst_ndarray(grads[nid], label="r") for nid, cl in clients.items()}
    creds = {"fusion_weight": weights, "label": "r", "n_chunks": clients["c0"].key_chunks(enc["c0"])}
    dks = {sid: kg.get_decryption_keys(sid, credentials=creds) for sid in sids}
    groups = []
    for enrolled in (sids[:2], sids[1:]):
        parts = {}
        for sid in enrolled:
            py, nat = _both(lambda: _server(pp, sid).compute_lst_ndarray_ct(enc, creds, dks[sid], enrolled))
            assert py == nat
            parts[sid] = nat
        groups.append(parts)
    assert native_calls["share_inputs"] == native_calls["share_partial"] == 4 * 4  # 4 chunks, 2 x 2 servers
    combiner = _server(pp, "comb")
    py, nat = _both(lambda: combiner.decrypt_lst_ndarray_ct(groups))
    assert native_calls["share_combine"] == 4 * 2
    expected = [sum(weights[nid][0] * grads[nid][l] for nid in clients) for l in range(2)]
    for l in range(2):
        assert np.array_equal(py[l], nat[l]) and np.array_equal(nat[l], expected[l])
    # a chunk's element 1 moved to bound + 1 (in c_y, which all partials share) does not decode
    q, delta = int(pp["q"]), int(pp["q"]) // int(pp["p"])
    bound = 3 * bound_x * bound_y
    mu = sum(weights[nid][0] * int(grads[nid][0][1]) for nid in clients)
    chunk = {}
    for sid, part in groups[0].items():
        c = dict(part[0]["parts"][0])
        c["c_y"] = list(c["c_y"])
        c["c_y"][1] = str((int(c["c_y"][1]) + delta * (bound + 1 - mu)) % q)
        chunk[sid] = c
    py, nat = _both(lambda: combiner.combine_decrypt(chunk))
    assert py == nat and nat[1] is None and None not in nat[:1] + nat[2:]


def _retag(kg, nid, label, ct):
    from pyfe4ai.schemes.mcfe.labels import encode_label
    from pyfe4ai.schemes.threshold_utils import tag_ciphertext

    return tag_ciphertext(dict(ct), kg.get_private_keys(nid)["mac_keys"], tmcfe_rlwe._TAG_DOMAIN,
                          tmcfe_rlwe._tag_payload(encode_label(label), ct))


@pytest.mark.parametrize("cfg", TCONFIGS[1:], ids=str)
def test_threshold_value_formats_and_errors_identical(cfg, native_calls):
    """Values the native reader does not take (blanks, signs, other digits, ...)
    go to the Python path; both give the same partials, and the same errors."""
    ring_n = cfg[0]
    kg, pp, clients, sids = _threshold(*cfg)
    weights = {nid: [1] for nid in clients}
    cts = {nid: cl.encrypt([[1, -2, 3]], "L") for nid, cl in clients.items()}
    creds = {"fusion_weight": weights, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=creds)
    server = _server(pp, "s0")
    q = int(pp["q"])

    def variant(f):
        ct = {"ct0": [list(cts["c1"]["ct0"][0])], "ct1": list(cts["c1"]["ct1"]), "k": cts["c1"]["k"],
              "label_hash": cts["c1"]["label_hash"]}
        f(ct)
        return dict(cts, c1=_retag(kg, "c1", "L", ct))

    share = lambda c: server.share_decrypt(c, creds, dk, sids[:2])
    reference = share(cts)
    variants = [  # (change, read natively, same values mod q)
        (lambda c: c["ct1"].__setitem__(0, int(c["ct1"][0])), True, True),            # exact ints
        (lambda c: c["ct1"].__setitem__(1, int(c["ct1"][1]) + 5 * q), True, True),    # any range
        (lambda c: c["ct1"].__setitem__(1, int(c["ct1"][1]) - q), False, True),       # below -2^63: Python's %
        (lambda c: c["ct1"].__setitem__(2, "00" + c["ct1"][2]), True, True),          # leading zeros
        (lambda c: c.__setitem__("ct0", (tuple(c["ct0"][0]),)), True, True),
        (lambda c: c["ct1"].__setitem__(0, " {} ".format(c["ct1"][0])), False, True),  # int() strips blanks
        (lambda c: c["ct1"].__setitem__(1, "+" + c["ct1"][1]), False, True),
        (lambda c: c["ct1"].__setitem__(4, "٣"), False, False),                       # non-ASCII digit
        (lambda c: c["ct1"].__setitem__(5, str(2 ** 300)), False, False),             # beyond 2^256
    ]
    for f, native, same in variants:
        native_calls.clear()
        py, nat = _both(lambda: share(variant(f)))
        assert py == nat
        assert (native_calls["share_partial"] == 1) == native
        assert (nat["w"] == reference["w"]) == same  # same values: the same noise
    bad_variants = [
        lambda c: c["ct1"].__setitem__(0, "12a"),
        lambda c: c["ct1"].__setitem__(0, 1.5),
        lambda c: c["ct1"].__setitem__(0, None),
        lambda c: c["ct1"].pop(),
        lambda c: c["ct0"].append(c["ct0"][0]),
        lambda c: c.__setitem__("k", ring_n + 1),
        lambda c: c.__setitem__("k", True),
        lambda c: c.__setitem__("ct0", "x"),
    ]
    for f in bad_variants:
        for name in ("python", "native"):
            with native_backend.backend(name), pytest.raises(FESchemeError, match="malformed ciphertext of c1"):
                share(variant(f))


class _BytesSystemRandom(random.Random):
    """``random.SystemRandom.getrandbits`` on a :class:`ByteStream` (big-endian
    bytes, as from os.urandom); randrange then draws as SystemRandom does."""

    def __init__(self, stream) -> None:
        self.stream = stream
        super().__init__(0)

    def getrandbits(self, k: int) -> int:
        nbytes = (k + 7) // 8
        return int.from_bytes(self.stream.take(nbytes), "big") >> (8 * nbytes - k)


@pytest.mark.parametrize("cfg", TCONFIGS, ids=str)
def test_threshold_key_shares_identical(cfg, monkeypatch):
    from pyfe4ai.schemes import threshold_utils

    kg, pp, clients, sids = _threshold(*cfg)
    creds = {"fusion_weight": {nid: [1] for nid in clients}, "label": "L", "n_chunks": 2}
    key = _mcfe_keys(kg, "s0", creds)
    with native_backend.backend("python"):
        monkeypatch.setattr(threshold_utils, "_CSPRNG", _BytesSystemRandom(ByteStream(6)))
        py = kg._share_key(key, "w")
    with native_backend.backend("native"):
        monkeypatch.setattr(native_backend, "_random_override", ByteStream(6).take)
        nat = kg._share_key(key, "w")
        monkeypatch.setattr(native_backend, "_random_override", None)
    assert py == nat
    with native_backend.backend("native"):  # the DRBG: shares still reconstruct the key
        shares = kg._share_key(key, "w")
    q = int(pp["q"])
    lam = {s: int(threshold_utils.lagrange_coefficient_at_zero(i + 1, [1, 3], q)) for i, s in enumerate(sids)}
    for poly, name in ((key["z"], "z"), (key["sk_y"]["c2"], None)):
        got = [(lam["s0"] * int(a) + lam["s2"] * int(b)) % q for a, b in zip(
            *[(shares[s][name] if name else shares[s]["sk_y"]["c2"]) for s in ("s0", "s2")])]
        assert got == [int(v) % q for v in poly]
    assert shares["s0"]["z"] != nat["s0"]["z"]


def test_threshold_standard_profile_native(native_calls):
    """mcfe/ring_lwe_threshold under the standard profile (q of 212 bits,
    ring_n 8192): encryption, keys, key shares, partials and combination native."""
    nids, sids = ["c0", "c1", "c2"], ["s0", "s1", "s2"]
    kg = tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator({
        "n": 3, "lst_nid": nids, "s": 3, "t": 2, "lst_sid": sids, "eta": 1, "bound_x": 100, "bound_y": 1,
        "security_profile": "standard"})
    kg.setup()
    pp = kg.get_public_parameters()
    assert int(pp["q"]).bit_length() > 200 and native_backend.ring_context(pp["q"], pp["ring_n"]).word_bits == 256
    rng = np.random.default_rng(2)
    grads = {nid: rng.uniform(-1, 1, 9000).round(2) for nid in nids}
    clients = {nid: tmcfe_rlwe.ThresholdMCFERingLWE({"id": nid, "precision": 2,
                                                    "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    enc = {nid: cl.encrypt_lst_ndarray([grads[nid]], label="r") for nid, cl in clients.items()}
    creds = {"fusion_weight": {nid: [1] for nid in nids}, "label": "r",
             "n_chunks": clients["c0"].key_chunks(enc["c0"])}
    parts = {}
    for sid in sids[1:]:
        dk = kg.get_decryption_keys(sid, credentials=creds)
        server = tmcfe_rlwe.ThresholdMCFERingLWE({"id": sid, "precision": 2, "keys": {"pp": pp}})
        parts[sid] = server.compute_lst_ndarray_ct(enc, creds, dk, sids[1:])
    combiner = tmcfe_rlwe.ThresholdMCFERingLWE({"id": "comb", "precision": 2, "keys": {"pp": pp}})
    out = combiner.decrypt_lst_ndarray_ct(parts)[0]
    assert np.allclose(out, sum(grads.values()))
    for name, count in (("encrypt_tensor", 3), ("pad_terms", 1), ("shamir_share", 1), ("share_partial", 4),
                        ("share_combine", 2)):
        assert native_calls[name] == count, name
