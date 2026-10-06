import os
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import DecentralizedMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import DecentralizedMCFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 1,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "ring_n": 16,
        "bound_x": 4,
        "bound_y": 3,
        "bound_u": 1,
        "label_modulus": 8,
    }


def test_key_generator(kg_config):
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_DMCFE_RING_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["n"] == kg_config["n"]
    assert pp["eta"] == kg_config["eta"]
    assert pp["ring_n"] == kg_config["ring_n"]

    sk = kg.get_private_keys("nid_0")
    assert "pk" in sk
    assert "u" in sk
    assert "sk" in sk
    assert "v" in sk


def test_entire_process(kg_config):
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()

    label = "dmcfe-ring-lwe"
    dct_x = {
        "nid_0": [[1, -2, 0, 3]],
        "nid_1": [[-1, 2, -2, 1]],
        "nid_2": [[2, 1, 1, -1]],
    }
    dct_y = {"nid_0": [1], "nid_1": [-1], "nid_2": [2]}
    expected = [0 for _ in range(len(dct_x["nid_0"][0]))]
    for nid in kg_config["lst_nid"]:
        for col in range(len(expected)):
            expected[col] += dct_x[nid][0][col] * dct_y[nid][0]

    pp = kg.get_public_parameters()
    dct_ct = {}
    dct_dk_shares = {}
    for nid in kg_config["lst_nid"]:
        crypto_nid = DecentralizedMCFERingLWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)

    decryptor = DecentralizedMCFERingLWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = decryptor.combine_function_decryption_key_share(dct_dk_shares)
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["bound_x"] = 40

    kg = DecentralizedMCFERingLWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "dmcfe-ring-lwe-ndarray"
    dct_x = {
        "nid_0": [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            np.array([[0.7, 0.8], [0.9, 1.0]]),
        ],
        "nid_1": [
            np.array([[0.2, 0.1, 0.4], [0.3, 0.6, 0.5]]),
            np.array([[0.5, 0.4], [0.3, 0.2]]),
        ],
        "nid_2": [
            np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        ],
    }
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}

    expected = []
    for layer_idx in range(len(dct_x[cfg["lst_nid"][0]])):
        expected.append(
            sum(
                (
                    (dct_x[nid][layer_idx] * pow(10, precision)).astype(int)
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
        crypto_nid = DecentralizedMCFERingLWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid], label=label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)

    decryptor = DecentralizedMCFERingLWE(
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
    kg = DecentralizedMCFERingLWEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})
