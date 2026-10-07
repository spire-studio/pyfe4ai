import os
import random

import pytest

pytest.importorskip("charm", reason="charm-crypto not installed")

from pyfe4ai.schemes.sife.part_fh_ipe_pairing import SIFEPartFHIPE
from pyfe4ai.schemes.sife.part_fh_ipe_pairing import SIFEPartFHIPEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 10,
        "bound": 10,
        "subspace_dim": 3,
        "coeff_bound": 1,
        "pairing_group_param": "SS512",
    }


def test_key_generator(kg_config):
    kg = SIFEPartFHIPEKeyGenerator(kg_config)

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE_PART_FH_IPE)
    assert os.path.exists(config_folder)
    assert os.path.exists(os.path.join(config_folder, "param.json"))

    kg.setup()
    pp = kg.get_public_parameters()
    assert pp["eta"] == kg_config["eta"]
    assert pp["subspace_dim"] == kg_config["subspace_dim"]

    sk = kg.get_private_keys()
    assert "b" in sk
    assert "u" in sk
    assert "v" in sk

    y = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "k" in dk
    assert len(dk["k"]) == kg_config["eta"] + 4


def test_public_and_secret_encryption(kg_config):
    kg = SIFEPartFHIPEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    sk = kg.get_private_keys()

    crypto = SIFEPartFHIPE({"id": "nid_default", "keys": {"pp": pp, "sk": sk}})
    decryptor = SIFEPartFHIPE({"id": "sid_0", "keys": {"pp": pp}})

    y = [random.randint(-kg_config["bound"], kg_config["bound"]) for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

    t = [random.randint(-kg_config["coeff_bound"], kg_config["coeff_bound"]) for _ in range(kg_config["subspace_dim"])]
    m = [[int(v) for v in row] for row in pp["m"]]
    x_from_subspace = [sum(m[i][j] * t[j] for j in range(len(t))) for i in range(len(m))]
    expected_subspace = sum(a * b for a, b in zip(x_from_subspace, y))
    ct_subspace = crypto.encrypt(t)
    computed_subspace = decryptor.decrypt(ct_subspace, dk)
    assert computed_subspace == expected_subspace

    x = [random.randint(-kg_config["bound"], kg_config["bound"]) for _ in range(kg_config["eta"])]
    expected_secret = sum(a * b for a, b in zip(x, y))
    ct_secret = crypto.sec_encrypt(x)
    computed_secret = decryptor.decrypt(ct_secret, dk)
    assert computed_secret == expected_secret
