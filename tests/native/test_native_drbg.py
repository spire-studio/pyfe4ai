"""The per-call DRBG of the Rust backend (see pyfe4ai.backend).

ChaCha20 streams are checked against an independent implementation (the
``cryptography`` package, if installed); with an injected seed, the native
encryption must equal the Python one fed the DRBG streams chunk by chunk,
and without one every call draws fresh randomness.
"""

from __future__ import annotations

import numpy as np
import os
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai import backend as native_backend
from pyfe4ai.backend import native_impl
from pyfe4ai.primitives import sampling as sampling_utils
from tests.native.test_native_backend import FixedBytes, StreamRandom, _rounded_sampler  # noqa: F401  (autouse)

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(native_backend.implementation() != "rust", reason="Rust extension not built"),
]
SEED = bytes(range(32))


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _stream(seed: bytes, stream: int, n: int) -> bytes:
    return native_backend.ext().drbg_bytes(seed, stream, n)


def test_streams_are_chacha20():
    algorithms = pytest.importorskip("cryptography.hazmat.primitives.ciphers.algorithms")
    from cryptography.hazmat.primitives.ciphers import Cipher

    for stream in (0, 1, 7, 2 ** 40):
        nonce = (0).to_bytes(8, "little") + stream.to_bytes(8, "little")  # 64-bit counter, 64-bit stream
        ref = Cipher(algorithms.ChaCha20(SEED, nonce), mode=None).encryptor().update(bytes(5000))
        assert _stream(SEED, stream, 5000) == ref
    assert _stream(SEED, 0, 64) != _stream(SEED, 1, 64) != _stream(bytes(32), 0, 64)


def _client(ring_n=32):
    nids = ["c0", "c1"]
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 2, "lst_nid": nids, "eta": 1, "ring_n": ring_n,
                                            "sec_param": 128, "bound_x": 100, "bound_y": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    return pp, mcfe_rlwe.MCFERingLWE({"id": "c0", "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys("c0")}})


@pytest.mark.parametrize("threads", [1, 3])
def test_seeded_encryption_reads_one_stream_per_chunk(threads, monkeypatch):
    pp, cl = _client()
    layers = [np.linspace(-1, 1, 70).round(2), np.array([0.5, -0.25])]
    per = 3 * native_impl.gauss_bytes(32)
    # one call per layer: the 70 values use streams 0-2, the second layer stream 0 of its own call
    expected_bytes = b"".join(_stream(SEED, c, per) for c in (0, 1, 2)) + _stream(SEED, 0, per)
    with native_backend.backend("python"):
        monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(FixedBytes(expected_bytes)))
        py = cl.encrypt_lst_ndarray(layers, label="r")
    monkeypatch.setattr(native_backend, "_seed_override", SEED)
    native_backend.set_threads(threads)
    try:
        with native_backend.backend("native"):
            nat = cl.encrypt_lst_ndarray(layers, label="r")
    finally:
        native_backend.set_threads(1)
    assert nat == py


def test_precompute_uses_the_same_streams(monkeypatch):
    pp, cl = _client()
    grad = np.linspace(-1, 1, 50).round(2)
    monkeypatch.setattr(native_backend, "_seed_override", SEED)
    with native_backend.backend("native"):
        direct = cl.encrypt_lst_ndarray([grad], label="r")
        pre = cl.precompute("r", [grad.shape])
        online = cl.encrypt_lst_ndarray([grad], label="r", precomputed=pre)
    assert online == direct


def test_single_matrix_encryption_reads_stream_zero(monkeypatch):
    pp, cl = _client()
    x = [[1, -2, 3]]
    rows = cl._padded_rows(x, extra_pad=cl._pad("L"))
    need = 3 * native_impl.gauss_bytes(32)
    with native_backend.backend("python"):
        monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(FixedBytes(_stream(SEED, 0, need))))
        py = cl._encrypt_rows(rows, 3)
    monkeypatch.setattr(native_backend, "_seed_override", SEED)
    with native_backend.backend("native"):
        assert cl._encrypt_rows(rows, 3) == py


def test_unseeded_calls_draw_fresh_randomness():
    pp, cl = _client()
    with native_backend.backend("native"):
        a = cl.encrypt_lst_ndarray([np.zeros(10)], label="a")[0]["chunks"][0]
        b = cl.encrypt_lst_ndarray([np.zeros(10)], label="a")[0]["chunks"][0]
    assert a["ct1"] != b["ct1"]
