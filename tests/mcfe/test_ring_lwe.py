import os
import random

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "ring_n": 16,
        "bound_x": 2,
        "bound_y": 2,
        "bound_u": 1,
        "label_modulus": 8,
    }


def test_key_generator(kg_config):
    kg = MCFERingLWEKeyGenerator(kg_config)

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_MCFE_RING_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["eta"] == kg_config["eta"]
    assert pp["n"] == kg_config["n"]
    assert pp["ring_n"] == kg_config["ring_n"]
    assert pp["label_modulus"] == kg_config["label_modulus"]

    sk = kg.get_private_keys("nid_0")
    assert "pk" in sk
    assert "u" in sk
    assert len(sk["pk"]) == kg_config["eta"]
    assert len(sk["u"]) == kg_config["eta"]

    y = {"nid_0": [1, -1], "nid_1": [0, 1], "nid_2": [1, 1]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "sk_y" in dk
    assert "z" in dk
    assert set(dk["sk_y"].keys()) == set(kg_config["lst_nid"])


def test_entire_process(kg_config):
    kg = MCFERingLWEKeyGenerator(kg_config)
    kg.setup()

    cols = 4
    label = "mcfe-ring-lwe"
    dct_x = {
        nid: [
            [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(cols)]
            for _ in range(kg_config["eta"])
        ]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(kg_config["eta"])]
        for nid in kg_config["lst_nid"]
    }

    expected = [0 for _ in range(cols)]
    for nid in kg_config["lst_nid"]:
        for col in range(cols):
            expected[col] += sum(
                dct_x[nid][row][col] * dct_y[nid][row] for row in range(kg_config["eta"])
            )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in kg_config["lst_nid"]:
        encryptor = MCFERingLWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid], label)

    decryptor = MCFERingLWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y, label)

    assert computed == expected


def test_entire_process_lst_ndarray():
    cfg = {
        "sec_param": 64,
        "eta": 1,
        "n": 2,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1"],
        "ring_n": 16,
        "bound_x": 30,
        "bound_y": 2,
        "bound_u": 1,
        "label_modulus": 8,
    }
    precision = 1
    label = "mcfe-ring-lwe-ndarray"

    kg = MCFERingLWEKeyGenerator(cfg)
    kg.setup()

    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2))]
        for nid in cfg["lst_nid"]
    }
    dct_y = {"nid_0": [1], "nid_1": [-1]}

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
    for nid in cfg["lst_nid"]:
        encryptor = MCFERingLWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = encryptor.encrypt_lst_ndarray(dct_x[nid], label=label)

    decryptor = MCFERingLWE(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(
        dct_ct, dk=dk, fusion_weight=dct_y, label=label
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)
