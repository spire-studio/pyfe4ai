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
        "pairing_group_param": "MNT224",
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
    assert len(sk["b_star"]) == kg_config["eta"] + 2  # padded (x, 1, 0)

    y = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "k1" in dk
    assert "k2" in dk
    assert len(dk["k2"]) == kg_config["eta"] + 2


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


@pytest.mark.parametrize("eta", [1, 3])
def test_padding_hides_x_and_zero_vectors(eta):
    """With eta = 1 (B* = 1) ciphertexts revealed x; all-zero vectors were
    recognisable. Vectors are now padded to (x, 1, 0) / (y, 0, 1)."""
    from pyfe4ai.schemes.sife.fh_ipe_pairing import SIFEFHIPE, SIFEFHIPEKeyGenerator
    from pyfe4ai.utils.pairing_backend import ZR

    kg = SIFEFHIPEKeyGenerator({"sec_param": 64, "eta": eta, "bound_x": 50, "bound_y": 50,
                                "pairing_group_param": "MNT224"})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = SIFEFHIPE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    x = [49] + [0] * (eta - 1)
    ct = enc.encrypt(x)
    g = enc.group
    c1 = g.deserialize(ct["c1"].encode())
    c2 = [g.deserialize(v.encode()) for v in ct["c2"]]
    assert len(c2) == eta + 2
    assert all(c2[0] != c1 ** g.init(ZR, v) for v in range(-50, 51))
    one = c1 ** g.init(ZR, 0)
    assert not any(c == one for c in enc.encrypt([0] * eta)["c2"])
    zero_key = kg.get_decryption_keys(None, credentials={"fusion_weight": [0] * eta})
    assert SIFEFHIPE({"keys": {"pp": pp}}).decrypt(ct, zero_key, [0] * eta) == 0
    y = [-7] + [1] * (eta - 1)
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": y})
    assert SIFEFHIPE({"keys": {"pp": pp}}).decrypt(ct, dk, y) == -343


def test_decryption_does_not_need_the_function(kg_config):
    """The keys hide y, so the decryptor need not know it."""
    kg = SIFEFHIPEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    x = [1, -2, 3][: kg_config["eta"]] + [0] * max(0, kg_config["eta"] - 3)
    y = [2, 1, -1][: kg_config["eta"]] + [0] * max(0, kg_config["eta"] - 3)
    ct = SIFEFHIPE({"keys": {"pp": pp, "sk": kg.get_private_keys()}}).encrypt(x)
    dk = kg.get_decryption_keys("sid", credentials={"fusion_weight": y})
    assert SIFEFHIPE({"keys": {"pp": pp}}).decrypt(ct, dk) == sum(a * b for a, b in zip(x, y))
