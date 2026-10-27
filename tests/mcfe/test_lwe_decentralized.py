import logging
import os
import random

import numpy as np
import pytest

from tests.log_config import init_logging_config
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWE
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWEKeyGenerator
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.base import cache_dir

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 1,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lwe_n": 16,
        "bound_x": 6,
        "bound_y": 6,
        "bound_u": 2,
        "label_modulus": 8,
    }


def test_key_generator(kg_config):
    kg = DecentralizedMCFELWEKeyGenerator(kg_config)
    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.eta == kg_config["eta"]
    assert kg.lwe_n == kg_config["lwe_n"]

    config_folder = os.path.join(cache_dir(), "authority", CryptoCONST.TYPE_DMCFE_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["n"] == kg_config["n"]
    assert pp["eta"] == kg_config["eta"]
    assert pp["lwe_n"] == kg_config["lwe_n"]
    assert pp["lst_nid"] == kg_config["lst_nid"]

    sk = kg.get_private_keys("nid_0")
    assert set(sk) == {"z", "s2", "label_hash"}
    assert len(sk["s2"]) == kg.dim
    assert all(sum(col) == 0 for col in zip(*kg.msk["s2"].values()))


def test_entire_process(kg_config):
    kg = DecentralizedMCFELWEKeyGenerator(kg_config)
    kg.setup()

    label = "dmcfe-lwe-label"
    dct_x = {
        nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"])]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"])]
        for nid in kg_config["lst_nid"]
    }
    expected = sum(dct_x[nid][0] * dct_y[nid][0] for nid in kg_config["lst_nid"])

    pp = kg.get_public_parameters()
    dct_ct = {}
    dct_dk_shares = {}
    for nid in kg_config["lst_nid"]:
        crypto_nid = DecentralizedMCFELWE(
            {
                "id": nid,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)

    decryptor = DecentralizedMCFELWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["bound_x"] = 50

    kg = DecentralizedMCFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "dmcfe-lwe-ndarray"
    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
        for nid in cfg["lst_nid"]
    }
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}

    expected = []
    for layer_idx in range(len(dct_x[cfg["lst_nid"][0]])):
        expected.append(
            sum(
                (
                    np.rint(dct_x[nid][layer_idx] * pow(10, precision))
                    / pow(10, precision)
                )
                * dct_y[nid][0]
                for nid in cfg["lst_nid"]
            )
        )

    pp = kg.get_public_parameters()
    dct_ct = {}
    dct_dk_shares = {}
    for nid in cfg["lst_nid"]:
        crypto_nid = DecentralizedMCFELWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)

    decryptor = DecentralizedMCFELWE(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def test_get_decryption_keys_raises_not_implemented(kg_config):
    """Regression (N12): the method used to *return* NotImplementedError."""
    kg = DecentralizedMCFELWEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})


def test_combined_key_equals_the_central_key(kg_config):
    """Fig. 14 yields exactly the LWE MCFE key sum y_i Z_i mod q."""
    from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator

    cfg = dict(kg_config, eta=2)
    kg = DecentralizedMCFELWEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_y = {nid: [random.randint(-6, 6), random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    shares = {
        nid: DecentralizedMCFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        .derive_function_decryption_key_share(dct_y)
        for nid in cfg["lst_nid"]
    }
    dk = DecentralizedMCFELWE({"id": "agg", "keys": {"pp": pp}}).combine_function_decryption_key_share(shares)
    central = MCFELWEKeyGenerator._aggregate_key(kg, dct_y)
    assert [int(v) for v in dk["sk_y"]] == [int(v) for v in central]
