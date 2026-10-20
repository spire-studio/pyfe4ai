"""The binary ciphertext format of the Ring-LWE MIFE / MCFE (``ct_format="binary"``).

Both backends write the same bytes (from the same Gaussian samples) and read
both formats; malformed binary ciphertexts are refused like malformed JSON
ones.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe_rlwe
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives import sampling as sampling_utils
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.ring_lwe_utils import decode_binary_ciphertext, encode_binary_ciphertext
from tests.test_native_backend import _rounded_sampler  # noqa: F401  (autouse)

pytestmark = pytest.mark.core
BACKENDS = ["python"] + (["native"] if native_backend.available() else [])


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _setup(ct_format="binary", eta=1, ring_n=16, bound_x=100, bound_y=1, precision=2):
    nids = ["c0", "c1", "c2"]
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": nids, "eta": eta, "ring_n": ring_n,
                                            "sec_param": 128, "bound_x": bound_x, "bound_y": bound_y})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: mcfe_rlwe.MCFERingLWE({"id": nid, "precision": precision, "ct_format": ct_format,
                                           "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in nids}
    return kg, pp, clients, mcfe_rlwe.MCFERingLWE({"id": "a", "precision": precision, "keys": {"pp": pp}})


def test_codec_roundtrip_and_header():
    q, n = 2 ** 77 + 0x4001, 16  # any modulus: the codec does not need a prime
    rng = random.Random(1)
    ct0 = [[rng.randrange(q) for _ in range(n)] for _ in range(2)]
    ct1 = [rng.randrange(q) for _ in range(n)]
    data = encode_binary_ciphertext(ct0, ct1, q, n)
    assert data[:6] == b"PFRL\x01\x0a" and len(data) == 10 + 3 * n * 10
    assert decode_binary_ciphertext(data, q, 2, n) == (ct0, ct1)
    for bad in (data[:-1], data + b"\0", b"XFRL" + data[4:], data[:4] + b"\x02" + data[5:]):
        with pytest.raises(ValueError):
            decode_binary_ciphertext(bad, q, 2, n)
    with pytest.raises(ValueError):
        decode_binary_ciphertext(data, q, 1, n)


@pytest.mark.skipif(not native_backend.available(), reason="native extension not built")
@pytest.mark.parametrize("eta", [1, 2])
def test_backends_write_the_same_bytes(eta, monkeypatch):
    from tests.test_native_backend import ByteStream, StreamRandom

    _, _, clients, _ = _setup(eta=eta, ring_n=32, precision=0, bound_x=3, bound_y=3)
    cl = clients["c0"]
    out = {}
    for name in BACKENDS:
        with native_backend.backend(name):
            monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(3)))
            monkeypatch.setattr(native_backend, "_random_override", ByteStream(3).take)
            if eta == 1:
                out[name] = cl.encrypt_lst_ndarray([np.arange(70) % 7 - 3], label="r")
            else:
                out[name] = cl.encrypt([[1, -2, 3], [0, 3, -3]], "r")
    assert out["python"] == out["native"]


@pytest.mark.parametrize("enc_backend", BACKENDS)
@pytest.mark.parametrize("dec_backend", BACKENDS)
def test_binary_and_mixed_formats_decrypt(enc_backend, dec_backend):
    kg, pp, clients, agg = _setup()
    json_client = mcfe_rlwe.MCFERingLWE({"id": "c2", "precision": 2,
                                         "keys": {"pp": pp, "sk": kg.get_private_keys("c2")}})
    rng = np.random.default_rng(2)
    grads = {nid: [rng.uniform(-1, 1, 37).round(2)] for nid in clients}
    with native_backend.backend(enc_backend):
        enc = {nid: (json_client if nid == "c2" else cl).encrypt_lst_ndarray(grads[nid], label="r")
               for nid, cl in clients.items()}
    assert isinstance(enc["c0"][0]["chunks"][0]["ct"], bytes) and "ct0" in enc["c2"][0]["chunks"][0]
    weights = {nid: [1] for nid in clients}
    dk = kg.get_decryption_keys("a", credentials={"fusion_weight": weights, "label": "r",
                                                  "n_chunks": agg.key_chunks(enc["c0"])})
    with native_backend.backend(dec_backend):
        out = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="r")
    assert np.allclose(out[0], sum(g[0] for g in grads.values()))
    # malformed binary ciphertexts
    for bad in (b"", enc["c0"][0]["chunks"][1]["ct"][:-1], "not bytes", 5):
        tampered = {nid: [dict(layer, chunks=list(layer["chunks"])) for layer in enc[nid]] for nid in enc}
        tampered["c1"][0]["chunks"][1] = {"ct": bad, "k": tampered["c1"][0]["chunks"][1]["k"]}
        with native_backend.backend(dec_backend), pytest.raises(FESchemeError, match="malformed ciphertext of c1"):
            agg.decrypt_lst_ndarray_ct(tampered, dk=dk, fusion_weight=weights, label="r")


def test_formats_are_validated():
    with pytest.raises(FEValidationError):
        _setup(ct_format="xml")
    nids, sids = ["c0", "c1"], ["s0", "s1"]
    kg = tmcfe_rlwe.ThresholdMCFERingLWEKeyGenerator({
        "n": 2, "lst_nid": nids, "s": 2, "t": 2, "lst_sid": sids, "eta": 1, "ring_n": 16,
        "sec_param": 64, "bound_x": 2, "bound_y": 2, "ct_format": "json"})
    kg.setup()
    with pytest.raises(FEValidationError):  # JSON public parameters: format 2 needs ct_format="binary" at the key generator
        tmcfe_rlwe.ThresholdMCFERingLWE({"id": "c0", "ct_format": "binary",
                                         "keys": {"pp": kg.get_public_parameters()}})


@pytest.mark.parametrize("name", BACKENDS)
def test_ml_adapter_binary_round(name):
    from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    nids = ["c0", "c1", "c2"]
    with native_backend.backend(name):
        w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": nids, "ring_n": 64, "sec_param": 64,
                                                 "bound_x": 100, "bound_y": 1}, precision=2,
                            crypto_options={"ct_format": "binary"})
        g = {n: np.linspace(-1, 1, 150).round(2) * (i + 1) / 3 for i, n in enumerate(nids)}
        enc = {n: encrypt_gradient(g[n], w, nid=n, label="r") for n in nids}
        out = aggregate_gradients(enc, w, {n: 1 for n in nids}, label="r")
    assert np.allclose(out, sum(g.values()))
    sample = enc["c0"].ciphertexts[0]["chunks"][0]
    assert set(sample) == {"ct", "k"} and isinstance(sample["ct"], bytes)
