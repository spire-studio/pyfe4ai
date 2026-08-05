import os
import random

import pytest

from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 4,
        "ring_n": 16,
        "bound_x": 2,
        "bound_y": 2,
    }


def test_key_generator(kg_config):
    kg = SIFERingLWEKeyGenerator(kg_config)

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE_RING_LWE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["eta"] == kg_config["eta"]
    assert pp["ring_n"] == kg_config["ring_n"]
    assert len(pp["A"]) == kg_config["ring_n"]

    sk = kg.get_private_keys()
    assert "pk" in sk
    assert len(sk["pk"]) == kg_config["eta"]

    y = [1, -1, 2, 0]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "sk_y" in dk
    assert len(dk["sk_y"]) == kg_config["ring_n"]


def test_entire_process(kg_config):
    kg = SIFERingLWEKeyGenerator(kg_config)
    kg.setup()

    cols = 3
    x = [
        [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(cols)]
        for _ in range(kg_config["eta"])
    ]
    y = [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(kg_config["eta"])]

    expected = []
    for col in range(cols):
        expected.append(sum(x[row][col] * y[row] for row in range(kg_config["eta"])))

    pp = kg.get_public_parameters()
    encryptor = SIFERingLWE({"id": "nid_default", "keys": {"pp": pp, "sk": kg.get_private_keys()}})
    decryptor = SIFERingLWE({"id": "sid_0", "keys": {"pp": pp}})

    ct = encryptor.encrypt(x)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    computed = decryptor.decrypt(ct, dk, y)

    assert computed == expected
