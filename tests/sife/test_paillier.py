import os
import random
import logging
import pytest

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.sife.paillier import SIFEPaillier
from pyfe4ai.schemes.sife.paillier import SIFEPaillierKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 4,
        "bit_length": 64,
        "bound_x": 10,
        "bound_y": 10,
    }


def test_key_generator(kg_config):
    kg = SIFEPaillierKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]
    assert kg.bit_length == kg_config["bit_length"]

    config_folder = os.path.join(
        "config", "authority", CryptoCONST.TYPE_SIFE_PAILLIER
    )
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.n) == pp["n"]
    assert gp.digits(kg.g) == pp["g"]

    sk_nid = kg.get_private_keys()
    assert "pk" in sk_nid

    fusion_weight = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "k" in dk
    assert isinstance(dk["k"], str)


def test_entire_process(kg_config):
    kg = SIFEPaillierKeyGenerator(kg_config)
    kg.setup()

    x = [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(4)]
    y = [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(4)]
    expected_inner_prod = sum(a * b for a, b in zip(x, y))

    pp = kg.get_public_parameters()
    sk_nid = kg.get_private_keys()
    config_nid = {
        "id": "nid_default",
        "keys": {"pp": pp, "sk": sk_nid},
    }
    crypto_nid = SIFEPaillier(config_nid)
    ct_nid = crypto_nid.encrypt(x)
    assert isinstance(ct_nid, dict)
    assert "ct0" in ct_nid
    assert "ct1" in ct_nid

    config_sid = {
        "id": "sid_0",
        "keys": {"pp": pp},
    }
    crypto_sid = SIFEPaillier(config_sid)
    dk = kg.get_decryption_keys(config_sid["id"], credentials={"fusion_weight": y})
    computed_inner_prod = crypto_sid.decrypt(ct_nid, dk, y)
    logger.info("<{},{}> expected inner-prod = {}".format(x, y, expected_inner_prod))
    logger.info("<{},{}> computed inner-prod = {}".format(x, y, computed_inner_prod))
    assert computed_inner_prod == expected_inner_prod


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = SIFEPaillierKeyGenerator(cfg)
    kg.setup()

    precision = 1
    fusion_weight = [2]
    lst_x = [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
    expected = [
        ((ary * pow(10, precision)).astype(int) / pow(10, precision))
        * fusion_weight[0]
        for ary in lst_x
    ]

    pp = kg.get_public_parameters()
    crypto_nid = SIFEPaillier(
        {
            "id": "nid_default",
            "precision": precision,
            "keys": {"pp": pp, "sk": kg.get_private_keys()},
        }
    )
    crypto_sid = SIFEPaillier(
        {
            "id": "sid_0",
            "precision": precision,
            "keys": {"pp": pp},
        }
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": fusion_weight})
    ct_nid = crypto_nid.encrypt_lst_ndarray(lst_x)
    computed = crypto_sid.compute_lst_ndarray_ct(
        ct_nid, dk=dk, fusion_weight=fusion_weight
    )

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)
