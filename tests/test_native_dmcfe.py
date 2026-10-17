"""Native key shares of the decentralized Ring-LWE MCFE against the Python code."""

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_decentralized as dm
from pyfe4ai.utils import native_backend

pytestmark = pytest.mark.skipif(not native_backend.available(), reason="Rust extension not built")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with native_backend.backend("auto"):
        yield


def _setup(eta, ring_n, n=3):
    nids = ["c{}".format(i) for i in range(n)]
    kg = dm.DecentralizedMCFERingLWEKeyGenerator(
        {"n": n, "lst_nid": nids, "eta": eta, "ring_n": ring_n, "sec_param": 64, "bound_x": 50, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: dm.DecentralizedMCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in nids}
    return pp, clients, dm.DecentralizedMCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": pp}})


@pytest.mark.parametrize("eta,ring_n,n,n_chunks", [(1, 16, 2, 0), (1, 64, 3, 3), (2, 32, 4, 2)])
def test_key_shares_and_combination_identical(eta, ring_n, n, n_chunks):
    pp, clients, agg = _setup(eta, ring_n, n)
    rng = np.random.default_rng(eta + n)
    weights = {nid: [int(v) for v in rng.integers(-3, 4, eta)] for nid in clients}
    shares = {}
    for name in ("python", "native"):
        with native_backend.backend(name):
            shares[name] = {nid: c.derive_function_decryption_key_share(weights, "lab", n_chunks)
                            for nid, c in clients.items()}
    assert shares["python"] == shares["native"]
    keys = {}
    for name in ("python", "native"):
        with native_backend.backend(name):
            keys[name] = agg.combine_function_decryption_key_share(shares["python"])
    assert keys["python"] == keys["native"]


def test_native_shares_decrypt():
    pp, clients, agg = _setup(1, 64, 3)
    rng = np.random.default_rng(1)
    xs = {nid: rng.integers(-50, 51, 150) for nid in clients}
    weights = {nid: [1] for nid in clients}
    with native_backend.backend("native"):
        enc = {nid: c.encrypt_lst_ndarray([xs[nid]], label="L") for nid, c in clients.items()}
        n_chunks = dm.DecentralizedMCFERingLWE.key_chunks(enc["c0"])
        shares = {nid: c.derive_function_decryption_key_share(weights, "L", n_chunks) for nid, c in clients.items()}
        dk = agg.combine_function_decryption_key_share(shares)
        out = agg.compute_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="L")
    assert np.array_equal(np.asarray(out[0]), sum(xs.values()))
