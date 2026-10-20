import os
import random

import pytest

from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.schemes.sife.part_fh_ipe_pairing import SIFEPartFHIPE
from pyfe4ai.schemes.sife.part_fh_ipe_pairing import SIFEPartFHIPEKeyGenerator
from pyfe4ai.core.constants import CryptoCONST


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 10,
        "bound": 10,
        "subspace_dim": 3,
        "coeff_bound": 1,
        "pairing_group_param": GROUP,
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

    # public-key scheme: the master secret key (b, U, V) is never handed out
    assert kg.get_private_keys() is None

    y = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    assert "k" in dk
    assert len(dk["k"]) == kg_config["eta"] + 4


def test_public_and_secret_encryption(kg_config):
    kg = SIFEPartFHIPEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    crypto = SIFEPartFHIPE({"id": "nid_default", "keys": {"pp": pp}})
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
    ct_secret = kg.sec_encrypt(x)
    computed_secret = decryptor.decrypt(ct_secret, dk)
    assert computed_secret == expected_secret


def test_subspace_dimension_must_be_proper(kg_config):
    """A square M (subspace_dim = eta) published V = (V^T M) M^-1 and with it y."""
    from pyfe4ai.core.exceptions import FEValidationError

    with pytest.raises(FEValidationError):
        SIFEPartFHIPEKeyGenerator(dict(kg_config, subspace_dim=kg_config["eta"]))
    default_cfg = {k: v for k, v in kg_config.items() if k != "subspace_dim"}
    # keys leak M^T y: the default keeps k = 1
    assert SIFEPartFHIPEKeyGenerator(dict(default_cfg, eta=3)).subspace_dim == 1


def test_adapter_encryptor_gets_no_master_key():
    from pyfe4ai.ml import FESchemeWrapper

    wrapper = FESchemeWrapper("sife", "part_fh_ipe_pairing",
                              {"sec_param": 64, "eta": 4, "bound": 4, "pairing_group_param": GROUP})
    assert "sk" not in wrapper._get_encryptor("nid_0").keys
