import os
import random

import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.sife.fh_ipe_pairing import SIFEFHIPE
from pyfe4ai.schemes.sife.fh_ipe_pairing import SIFEFHIPEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 8,
        "bound_x": 8,
        "bound_y": 8,
        "pairing_group_param": "SS512",
    }


def test_key_generator(kg_config):
    kg = SIFEFHIPEKeyGenerator(kg_config)

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE_FH_IPE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["eta"] == kg_config["eta"]
    assert pp["pairing_group_param"] == kg_config["pairing_group_param"]

    sk = kg.get_private_keys()
    assert "g2" in sk
    assert "b_star" in sk
    assert len(sk["b_star"]) == kg_config["eta"]

    y = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "k1" in dk
    assert "k2" in dk
    assert len(dk["k2"]) == kg_config["eta"]


def test_entire_process(kg_config):
    kg = SIFEFHIPEKeyGenerator(kg_config)
    kg.setup()

    x = [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(kg_config["eta"])]
    y = [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(kg_config["eta"])]
    expected = sum(a * b for a, b in zip(x, y))

    pp = kg.get_public_parameters()
    encryptor = SIFEFHIPE({"id": "nid_default", "keys": {"pp": pp, "sk": kg.get_private_keys()}})
    decryptor = SIFEFHIPE({"id": "sid_0", "keys": {"pp": pp}})

    ct = encryptor.encrypt(x)
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    computed = decryptor.decrypt(ct, dk, y)
    assert computed == expected
