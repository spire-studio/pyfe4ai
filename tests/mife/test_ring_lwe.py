import os
import random

import pytest

from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 3,
        "n": 2,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1"],
        "ring_n": 16,
        "bound_x": 2,
        "bound_y": 2,
    }


def test_key_generator(kg_config):
    kg = MIFERingLWEKeyGenerator(kg_config)

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_MIFE_RING_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["eta"] == kg_config["eta"]
    assert pp["n"] == kg_config["n"]
    assert pp["ring_n"] == kg_config["ring_n"]

    sk = kg.get_private_keys("nid_0")
    assert "pk" in sk
    assert len(sk["pk"]) == kg_config["eta"]

    y = {"nid_0": [1, -1, 2], "nid_1": [0, 1, -1]}
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "sk_y" in dk
    assert set(dk["sk_y"].keys()) == set(kg_config["lst_nid"])


def test_entire_process(kg_config):
    kg = MIFERingLWEKeyGenerator(kg_config)
    kg.setup()

    cols = 4
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
        encryptor = MIFERingLWE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = encryptor.encrypt(dct_x[nid])

    decryptor = MIFERingLWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y)

    assert computed == expected
