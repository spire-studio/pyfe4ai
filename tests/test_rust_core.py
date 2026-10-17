"""The Rust core (rust/core) computes exactly what the Python code computes.

Loads the core's C ABI (src/capi.rs) with ctypes from ``PYFE4AI_RUST_CORE_LIB``
or ``rust/core/target/release``; skipped when it is not built
(``cargo build --release`` in rust/core). The same comparisons as
tests/test_native_backend.py: NTT, sampler, SHAKE-256, hash to Z_q, Ring-LWE
encryption, decryption and key terms, dense-LWE dot products.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import random
from pathlib import Path

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import lwe as mcfe_lwe
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils import sampling_utils
from pyfe4ai.utils.crypto_utils import hash_to_zq_material, hash_to_zq_vector
from tests.test_native_backend import ByteStream, FixedBytes, StreamRandom, _rounded_sampler  # noqa: F401

pytestmark = pytest.mark.core


def _find_lib():
    names = ["libpyfe4ai_core.dylib", "libpyfe4ai_core.so", "pyfe4ai_core.dll"]
    roots = [os.environ.get("PYFE4AI_RUST_CORE_LIB", "")]
    root = Path(__file__).resolve().parents[1] / "rust" / "core" / "target" / "release"
    roots += [str(root / name) for name in names]
    if os.environ.get("CARGO_TARGET_DIR"):
        roots += [str(Path(os.environ["CARGO_TARGET_DIR"]) / "release" / name) for name in names]
    for path in roots:
        if path and os.path.exists(path):
            return ctypes.CDLL(path)
    return None


LIB = _find_lib()
pytestmark = [pytest.mark.core, pytest.mark.skipif(LIB is None, reason="rust/core not built")]
if LIB is not None:
    LIB.pf4_ring_new.restype = ctypes.c_void_p
    LIB.pf4_ring_new.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    LIB.pf4_ring_free.argtypes = [ctypes.c_void_p]
    LIB.pf4_word_bits.argtypes = [ctypes.c_void_p]
    LIB.pf4_ntt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    LIB.pf4_gauss.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double,
                              ctypes.POINTER(ctypes.c_int64)]
    LIB.pf4_shake256.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    LIB.pf4_hash_to_zq.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
                                   ctypes.c_char_p]
    LIB.pf4_encrypt_rows.argtypes = [ctypes.c_void_p] + [ctypes.c_char_p] * 2 + [ctypes.c_size_t] + \
        [ctypes.c_char_p] * 2 + [ctypes.c_double] * 2 + [ctypes.c_char_p, ctypes.c_size_t] + [ctypes.c_char_p] * 2
    LIB.pf4_encrypt_tensor.restype = ctypes.c_int64
    LIB.pf4_encrypt_tensor.argtypes = [ctypes.c_void_p] + [ctypes.c_char_p] * 4 + [
        ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p]
    LIB.pf4_decrypt_chunk.restype = ctypes.c_int
    LIB.pf4_decrypt_chunk.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t] + [ctypes.c_char_p] * 5 + [
        ctypes.c_int64, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64), ctypes.c_char_p]
    LIB.pf4_pad_terms.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
                                  ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p]
    LIB.pf4_xmd64.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    LIB.pf4_dense_dots.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t),
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_uint64)]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def pack(values, q=None) -> bytes:
    return b"".join(((int(v) % int(q)) if q is not None else int(v)).to_bytes(16, "little") for v in values)


def unpack(data: bytes) -> list:
    return [int.from_bytes(data[i:i + 16], "little") for i in range(0, len(data), 16)]


class Ring:
    def __init__(self, q, n):
        self.q, self.n = int(q), n
        root = rlu._get_ntt_root(gp.mpz(q), n)
        self.ptr = LIB.pf4_ring_new(pack([q]), n, pack([root]))
        assert self.ptr

    def __del__(self):
        LIB.pf4_ring_free(self.ptr)


def _materials(mats):
    arr = (ctypes.c_char_p * len(mats))(*mats)
    lens = (ctypes.c_size_t * len(mats))(*[len(m) for m in mats])
    return arr, lens


# ------------------------------------------------------------------ kernels

@pytest.mark.parametrize("bits,n", [(20, 2), (40, 16), (61, 256), (62, 1024), (63, 64), (77, 1024), (124, 512)])
def test_ntt_identical(bits, n):
    q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), n)
    ring = Ring(q, n)
    assert LIB.pf4_word_bits(ring.ptr) == (64 if q < 2 ** 62 else 128)
    root = rlu._get_ntt_root(q, n)
    rng = random.Random(bits)
    a = [gp.mpz(rng.randrange(int(q))) for _ in range(n)]
    out = ctypes.create_string_buffer(16 * n)
    LIB.pf4_ntt(ring.ptr, pack(a), out, 0)
    assert unpack(out.raw) == [int(v) for v in rlu.ntt_forward(a, q, root)]
    LIB.pf4_ntt(ring.ptr, pack(a), out, 1)
    assert unpack(out.raw) == [int(v) for v in rlu.ntt_inverse(a, q, root)]


@pytest.mark.parametrize("sigma", [0.5, 3.2, 17.25, 4.97e9, 2.0 ** 40])
@pytest.mark.parametrize("count", [1, 2, 7, 1024])
def test_gaussian_identical(sigma, count):
    data = random.Random(count).randbytes(native_backend.gauss_bytes(count))
    ref = StreamRandom(FixedBytes(data))
    expected = [int(round(ref.gauss(0.0, sigma))) for _ in range(count)]
    out = (ctypes.c_int64 * count)()
    LIB.pf4_gauss(data, len(data), count, sigma, out)
    assert list(out) == expected


@pytest.mark.parametrize("length", [0, 1, 135, 136, 137, 272, 5000])
def test_shake256_identical(length):
    msg = random.Random(length).randbytes(length)
    for out_len in (1, 136, 137, 4097):
        out = ctypes.create_string_buffer(out_len)
        LIB.pf4_shake256(msg, len(msg), out, out_len)
        assert out.raw == hashlib.shake_256(msg).digest(out_len)


@pytest.mark.parametrize("bits", [40, 61, 62, 64, 77, 124])
def test_hash_to_zq_identical(bits):
    q = rlu.next_ntt_prime(gp.mpz(2) ** (bits - 1), 64)
    ring = Ring(q, 64)
    mat = hash_to_zq_material("u'|label", "pyfe4ai/mcfe/ring_lwe/H(u',label)")
    out = ctypes.create_string_buffer(16 * 300)
    LIB.pf4_hash_to_zq(ring.ptr, mat, len(mat), 300, out)
    assert unpack(out.raw) == [int(v) for v in hash_to_zq_vector(
        "u'|label", q, 300, domain="pyfe4ai/mcfe/ring_lwe/H(u',label)")]


def test_expand_message_xmd_identical():
    from pyfe4ai.utils.ristretto255 import expand_message_xmd

    rng = random.Random(8)
    for length in (0, 1, 111, 112, 127, 128, 129, 300):
        msg, dst = rng.randbytes(length), rng.randbytes(rng.randrange(1, 256))
        out = ctypes.create_string_buffer(64)
        LIB.pf4_xmd64(msg, len(msg), dst, len(dst), out)
        assert out.raw == expand_message_xmd(msg, dst, 64)


def test_dense_dots_identical():
    q = gp.next_prime(gp.mpz(2) ** 186)
    count, kl, nb = 300, 3, (187 + 128 + 7) // 8
    rng = random.Random(5)
    key = [rng.randrange(int(q)) for _ in range(count)]
    labels = ["a", "b", "c"]
    mats = [hash_to_zq_material(lab, "d") for lab in labels]
    arr, lens = _materials(mats)
    sl = kl + (nb + 7) // 8 + 1
    out = (ctypes.c_uint64 * (sl * len(mats)))()
    key_limbs = (ctypes.c_uint64 * (count * kl))(*[(v >> (64 * i)) & (2 ** 64 - 1) for v in key for i in range(kl)])
    LIB.pf4_dense_dots(arr, lens, len(mats), nb, count, key_limbs, kl, 2, out)
    for m, lab in enumerate(labels):
        total = sum(int(out[m * sl + i]) << (64 * i) for i in range(sl))
        assert total % int(q) == int(mcfe_lwe.dot_mod(key, hash_to_zq_vector(lab, q, count, domain="d"), q))


# ------------------------------------------------------------- scheme paths

def _mcfe(ring_n=32, sec_param=128, bound_x=100, bound_y=1, eta=1):
    nids = ["c0", "c1", "c2"]
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": nids, "eta": eta, "ring_n": ring_n,
                                            "sec_param": sec_param, "bound_x": bound_x, "bound_y": bound_y})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: mcfe_rlwe.MCFERingLWE({"id": nid, "precision": 2,
                                           "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in nids}
    return kg, pp, clients, mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 2, "keys": {"pp": pp}})


@pytest.mark.parametrize("cfg", [(32, 64, 2, 2, 1), (16, 128, 100, 1, 2), (64, 128, 100, 1, 1)], ids=str)
def test_encrypt_rows_identical(cfg, monkeypatch):
    ring_n, sec, bx, by, eta = cfg
    kg, pp, clients, _ = _mcfe(ring_n, sec, bx, by, eta)
    cl = clients["c0"]
    q, n = int(pp["q"]), ring_n
    x = [[random.Random(r).randint(-bx, bx) for _ in range(n)] for r in range(eta)]
    w_rows = cl._padded_rows(x, extra_pad=cl._pad("L"))
    nbytes = native_backend.gauss_bytes(n) * (2 + eta)
    data = random.Random(1).randbytes(nbytes)
    with native_backend.backend("python"):
        monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(FixedBytes(data)))
        py0, py1 = cl._encrypt_rows_python(w_rows)
    ring = Ring(q, n)
    ct0, ct1 = ctypes.create_string_buffer(16 * eta * n), ctypes.create_string_buffer(16 * n)
    delta = q // int(pp["p"])
    LIB.pf4_encrypt_rows(ring.ptr, pack(pp["A"], q), pack([v for row in cl.sk["pk"] for v in row], q), eta,
                         pack([delta]), pack([v for row in w_rows for v in row]), float(pp["sigma2"]),
                         float(pp["sigma3"]), data, len(data), ct0, ct1)
    assert unpack(ct0.raw) == [int(v) for row in py0 for v in row]
    assert unpack(ct1.raw) == [int(v) for v in py1]


@pytest.mark.parametrize("size", [1, 32, 77])
@pytest.mark.parametrize("threads", [1, 3])
def test_encrypt_tensor_identical(size, threads, monkeypatch):
    kg, pp, clients, _ = _mcfe()
    cl = clients["c1"]
    rng = np.random.default_rng(size)
    layer = rng.uniform(-1, 1, size).round(2)
    with native_backend.backend("python"):
        monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(4)))
        py = cl.encrypt_lst_ndarray([layer], label="r")[0]["chunks"]
    q, n = int(pp["q"]), pp["ring_n"]
    ring = Ring(q, n)
    ints = np.ascontiguousarray(np.rint(layer * 100).astype(np.int64))
    chunks = -(-size // n)
    mats = [mcfe_rlwe.label_pad_material(cl.sk["u_prime"], mcfe_rlwe.chunk_label("r", c)) for c in range(chunks)]
    arr, lens = _materials(mats)
    data = ByteStream(4).take(3 * native_backend.gauss_bytes(n) * chunks)
    out = ctypes.create_string_buffer(32 * n * chunks)
    got = LIB.pf4_encrypt_tensor(
        ring.ptr, pack(pp["A"], q), pack(cl.sk["pk"][0], q), pack(cl.sk["u"][0], q), pack([q // int(pp["p"])]),
        ints.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), size, arr, lens, chunks, float(pp["sigma2"]),
        float(pp["sigma3"]), data, len(data), threads, out)
    assert got == chunks
    vals = unpack(out.raw)
    for j, ct in enumerate(py):
        assert [int(v) for v in ct["ct0"][0]] == vals[2 * n * j:2 * n * j + n]
        assert [int(v) for v in ct["ct1"]] == vals[2 * n * j + n:2 * n * (j + 1)]


def test_decrypt_chunk_identical():
    kg, pp, clients, agg = _mcfe(eta=2, ring_n=16)
    q, n, eta = int(pp["q"]), 16, 2
    rng = random.Random(3)
    weights = {nid: [rng.randint(-1, 1) or 1 for _ in range(eta)] for nid in clients}
    dk = kg.get_decryption_keys("a", credentials={"fusion_weight": weights, "label": "L"})
    ring = Ring(q, n)
    honest = {nid: cl.encrypt([[rng.randint(-100, 100) for _ in range(n)] for _ in range(eta)], "L")
              for nid, cl in clients.items()}
    randoms = {nid: {"ct0": [[str(rng.randrange(q)) for _ in range(n)] for _ in range(eta)],
                     "ct1": [str(rng.randrange(q)) for _ in range(n)], "k": n} for nid in clients}
    bound = 3 * eta * 100 * 1
    for cts in (honest, randoms):
        with native_backend.backend("python"):
            expected = agg._decrypt_with_pad_term(cts, dk["sk_y"], dk["z"], weights, n)
        nids = list(cts)
        slots = [int(v) for nid in nids for row in cts[nid]["ct0"] + [cts[nid]["ct1"]] for v in row]
        out = (ctypes.c_int64 * n)()
        none = ctypes.create_string_buffer(n)
        ok = LIB.pf4_decrypt_chunk(ring.ptr, eta, len(nids), pack(dk["z"], q), pack(slots, q),
                                   pack([v for nid in nids for v in weights[nid]], q),
                                   pack([v for nid in nids for v in dk["sk_y"][nid]], q),
                                   pack([q // int(pp["p"])]), bound, n, out, none)
        assert ok == 1
        got = [None if none.raw[i] else out[i] for i in range(n)]
        assert got == expected


def test_pad_terms_identical():
    kg, pp, clients, _ = _mcfe(ring_n=32, bound_y=2)
    weights = {nid: [2] for nid in clients}
    with native_backend.backend("python"):
        dk = kg.get_decryption_keys("a", credentials={"fusion_weight": weights, "label": "L", "n_chunks": 2})
    q, n = int(pp["q"]), 32
    labels = ["L", mcfe_rlwe.chunk_label("L", 0), mcfe_rlwe.chunk_label("L", 1)]
    mats = [mcfe_rlwe.label_pad_material(kg.msk["u_prime"][nid], lab) for lab in labels for nid in kg.lst_nid]
    arr, lens = _materials(mats)
    with native_backend.backend("python"):
        base = kg.__class__.__mro__[1].get_decryption_keys(kg, "a", credentials={"fusion_weight": weights})["z"]
    out = ctypes.create_string_buffer(16 * n * len(labels))
    ring = Ring(q, n)
    LIB.pf4_pad_terms(ring.ptr, pack(base, q), arr, lens, len(labels), 3, pack([2, 2, 2], q), 1, 2, out)
    vals = unpack(out.raw)
    assert [str(v) for v in vals[:n]] == dk["z"]
    assert [str(v) for v in vals[n:2 * n]] == dk["z_chunks"][0]
    assert [str(v) for v in vals[2 * n:]] == dk["z_chunks"][1]
