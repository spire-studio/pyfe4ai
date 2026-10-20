"""BMMS21 identity-based DMCFE over Ring-LWE (GDSum key derivation)."""

import random

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import DecentralizedMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import DecentralizedMCFERingLWEKeyGenerator
from pyfe4ai.core.exceptions import FESchemeError


@pytest.fixture
def kg_config():
    return {"sec_param": 64, "eta": 1, "n": 3, "lst_nid": ["nid_0", "nid_1", "nid_2"],
            "ring_n": 16, "bound_x": 4, "bound_y": 4}


def _clients(kg, pp, lst_nid, precision=0):
    return {
        nid: DecentralizedMCFERingLWE({"id": nid, "precision": precision, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        for nid in lst_nid
    }


def test_key_generator(kg_config):
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    sk = kg.get_private_keys("nid_0")
    assert set(sk) == {"pk", "u", "u_prime", "s", "pair_keys", "label_hash"}
    assert set(sk["pair_keys"]) == {"nid_1", "nid_2"}


def test_entire_process(kg_config):
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    lst_nid = kg_config["lst_nid"]
    clients = _clients(kg, pp, lst_nid)
    label = "dmcfe-ring"
    dct_x = {nid: [[random.randint(-4, 4) for _ in range(5)]] for nid in lst_nid}
    dct_y = {nid: [random.randint(-4, 4)] for nid in lst_nid}
    expected = [sum(dct_x[nid][0][c] * dct_y[nid][0] for nid in lst_nid) for c in range(5)]
    dct_ct = {nid: clients[nid].encrypt(dct_x[nid], label) for nid in lst_nid}
    shares = {nid: clients[nid].derive_function_decryption_key_share(dct_y, label) for nid in lst_nid}
    combiner = DecentralizedMCFERingLWE({"id": "agg", "keys": {"pp": pp}})
    dk = combiner.combine_function_decryption_key_share(shares)
    assert combiner.decrypt(dct_ct, dk, dct_y, label) == expected


def test_combined_key_equals_the_central_key(kg_config):
    """GDSum masks cancel: the combined key is the identity-based MCFE key."""
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = _clients(kg, pp, kg_config["lst_nid"])
    dct_y = {nid: [random.randint(-4, 4)] for nid in kg_config["lst_nid"]}
    shares = {nid: c.derive_function_decryption_key_share(dct_y, "lbl", n_chunks=2) for nid, c in clients.items()}
    dk = DecentralizedMCFERingLWE({"id": "agg", "keys": {"pp": pp}}).combine_function_decryption_key_share(shares)
    central = MCFERingLWEKeyGenerator.get_decryption_keys(
        kg, "sid", credentials={"fusion_weight": dct_y, "label": "lbl", "n_chunks": 2}
    )
    assert dk["z"] == central["z"] and dk["z_chunks"] == central["z_chunks"]
    assert dk["sk_y"] == central["sk_y"]
    # an individual share is masked
    assert shares["nid_0"]["z"] != central["z"]


def test_missing_or_mismatched_shares_rejected(kg_config):
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    clients = _clients(kg, pp, kg_config["lst_nid"])
    y = {nid: [1] for nid in kg_config["lst_nid"]}
    a = {nid: c.derive_function_decryption_key_share(y, "a") for nid, c in clients.items()}
    b = {nid: c.derive_function_decryption_key_share(y, "b") for nid, c in clients.items()}
    combiner = DecentralizedMCFERingLWE({"id": "agg", "keys": {"pp": pp}})
    with pytest.raises(FESchemeError):
        combiner.combine_function_decryption_key_share({k: a[k] for k in ["nid_0", "nid_1"]})
    with pytest.raises(FESchemeError):
        combiner.combine_function_decryption_key_share(dict(a, nid_2=b["nid_2"]))


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config, bound_x=50)
    kg = DecentralizedMCFERingLWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    precision, label = 1, "dmcfe-ring-ndarray"
    lst_nid = cfg["lst_nid"]
    clients = _clients(kg, pp, lst_nid, precision)
    dct_x = {nid: [np.random.random_sample((3, 7))] for nid in lst_nid}
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}
    expected = sum((np.rint(dct_x[nid][0] * 10) / 10) * dct_y[nid][0] for nid in lst_nid)
    dct_ct = {nid: clients[nid].encrypt_lst_ndarray(dct_x[nid], label=label) for nid in lst_nid}
    n_chunks = DecentralizedMCFERingLWE.key_chunks(dct_ct["nid_0"])
    shares = {nid: clients[nid].derive_function_decryption_key_share(dct_y, label, n_chunks) for nid in lst_nid}
    combiner = DecentralizedMCFERingLWE({"id": "agg", "precision": precision, "keys": {"pp": pp}})
    dk = combiner.combine_function_decryption_key_share(shares)
    computed = combiner.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y, label=label)
    assert np.allclose(computed[0], expected, atol=1e-2)


def test_get_decryption_keys_raises_not_implemented(kg_config):
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})
