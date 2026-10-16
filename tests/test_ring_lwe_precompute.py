"""Offline / online MCFE Ring-LWE: precompute(label, shapes), then encrypt.

The online phase must give the ciphertexts of a direct encryption with the
same Gaussian samples (both backends, both ciphertext formats), refuse a
second use, another label, client or shape, and the ML-adapter helpers must
aggregate correctly with a key prepared before the round.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import sampling_utils
from pyfe4ai.utils.exceptions import FEValidationError
from tests.test_native_backend import ByteStream, StreamRandom

pytestmark = pytest.mark.core
BACKENDS = ["python"] + (["native"] if native_backend.available() else [])


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _clients(ct_format="json", ring_n=16):
    nids = ["c0", "c1", "c2"]
    kg = mcfe_rlwe.MCFERingLWEKeyGenerator({"n": 3, "lst_nid": nids, "eta": 1, "ring_n": ring_n,
                                            "sec_param": 128, "bound_x": 100, "bound_y": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, {nid: mcfe_rlwe.MCFERingLWE({"id": nid, "precision": 2, "ct_format": ct_format,
                                            "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in nids}


def _seed(monkeypatch, seed):
    monkeypatch.setattr(sampling_utils, "_CSPRNG", StreamRandom(ByteStream(seed)))
    monkeypatch.setattr(native_backend, "_random_override", ByteStream(seed).take)


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("ct_format", ["json", "binary"])
def test_online_equals_direct(name, ct_format, monkeypatch):
    _, clients = _clients(ct_format)
    cl = clients["c0"]
    rng = np.random.default_rng(4)
    layers = [rng.uniform(-1, 1, 40).round(2), rng.uniform(-1, 1, (3, 5)).round(2), np.array(0.5)]
    with native_backend.backend(name):
        _seed(monkeypatch, 7)
        direct = cl.encrypt_lst_ndarray(layers, label="r")
        _seed(monkeypatch, 7)
        pre = cl.precompute("r", [a.shape for a in layers])
        monkeypatch.setattr(native_backend, "_random_override", None)
        online = cl.encrypt_lst_ndarray(layers, label="r", precomputed=pre)
    assert online == direct
    assert pre.used
    if name == "native":
        assert native_backend.ext().precomputed_info(pre.native[1]) == {"chunks": 3 + 1 + 1, "used": 5}


@pytest.mark.parametrize("name", BACKENDS)
def test_one_time_and_mismatches(name):
    _, clients = _clients()
    cl, other = clients["c0"], clients["c1"]
    grad = np.linspace(-1, 1, 20).round(2)
    with native_backend.backend(name):
        pre = cl.precompute("r", [grad.shape])
        with pytest.raises(FEValidationError, match="another label"):
            cl.encrypt_lst_ndarray([grad], label="s", precomputed=pre)
        with pytest.raises(FEValidationError, match="another client"):
            other.encrypt_lst_ndarray([grad], label="r", precomputed=pre)
        with pytest.raises(FEValidationError, match="shapes differ"):
            cl.encrypt_lst_ndarray([grad[:-1]], label="r", precomputed=pre)
        with pytest.raises(FEValidationError, match="exceeds the bound"):
            cl.encrypt_lst_ndarray([grad * 2], label="r", precomputed=pre)
        assert not pre.used  # refused inputs do not consume it
        cl.encrypt_lst_ndarray([grad], label="r", precomputed=pre)
        with pytest.raises(FEValidationError, match="used once"):
            cl.encrypt_lst_ndarray([grad], label="r", precomputed=pre)
        with pytest.raises(FEValidationError):
            cl.precompute("r", [(0,)])


@pytest.mark.parametrize("name", BACKENDS)
def test_ml_adapter_offline_online_round(name):
    from pyfe4ai.utils.ml_adapter import (
        FESchemeWrapper, aggregate_gradients, encrypt_gradient, precompute_gradient, prepare_aggregation,
    )

    nids = ["c0", "c1", "c2"]
    with native_backend.backend(name):
        w = FESchemeWrapper("mcfe", "ring_lwe", {"eta": 1, "n": 3, "lst_nid": nids, "ring_n": 64, "sec_param": 64,
                                                 "bound_x": 100, "bound_y": 2}, precision=2)
        shape = (5, 30)
        weights = {n: 1 for n in nids}
        # offline: client masks and the aggregator's key
        pre = {n: precompute_gradient(shape, w, nid=n, label="r1") for n in nids}
        prepared = prepare_aggregation(w, weights, shape=shape, label="r1")
        # online
        rng = np.random.default_rng(0)
        g = {n: rng.uniform(-1, 1, shape).round(2) for n in nids}
        enc = {n: encrypt_gradient(g[n], w, nid=n, label="r1", precomputed=pre[n]) for n in nids}
        out = aggregate_gradients(enc, w, label="r1", prepared=prepared)
        assert np.allclose(out, sum(g.values()))
        with pytest.raises(ValueError, match="other weights"):
            aggregate_gradients(enc, w, {n: 2 for n in nids}, label="r1", prepared=prepared)
        with pytest.raises(ValueError, match="for label"):
            aggregate_gradients(enc, w, label="r2", prepared=prepared)
        with pytest.raises(ValueError, match="already encrypted"):
            precompute_gradient(shape, w, nid="c0", label="r1")
        with pytest.raises(ValueError, match="other weights"):  # the label is bound to its weights
            prepare_aggregation(w, {n: 2 for n in nids}, shape=shape, label="r1")
        lw = FESchemeWrapper("mcfe", "ddh", {"sec_param": 128, "n": 3, "lst_nid": nids})
        with pytest.raises(ValueError, match="no offline"):
            precompute_gradient(shape, lw, nid="c0", label="r1")
