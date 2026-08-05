import os
import random
import logging
import pytest

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgard
from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgardKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture(params=[(4, 10), (3, 20)], ids=["eta=4,bound=10", "eta=3,bound=20"])
def kg_config(request):
    eta, bound = request.param
    return {
        "sec_param": 64,
        "eta": eta,
        "modulus_length": 64,
        "bound": bound,
    }


def test_key_generator(kg_config):
    kg = SIFEDamgardKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]
    assert kg.modulus_length == kg_config["modulus_length"]

    config_folder = os.path.join(
        "config", "authority", CryptoCONST.TYPE_SIFE_DAMGARD
    )
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.p) == pp["p"]
    assert gp.digits(kg.g) == pp["g"]
    assert gp.digits(kg.h) == pp["h"]

    sk_nid = kg.get_private_keys()
    assert "pk" in sk_nid

    fusion_weight = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "k1" in dk
    assert "k2" in dk


def test_entire_process(kg_config):
    kg = SIFEDamgardKeyGenerator(kg_config)
    kg.setup()

    eta = kg_config["eta"]
    bound = kg_config["bound"]
    x = [random.randint(-bound, bound) for _ in range(eta)]
    y = [random.randint(-bound, bound) for _ in range(eta)]
    expected_inner_prod = sum(a * b for a, b in zip(x, y))

    pp = kg.get_public_parameters()
    sk_nid = kg.get_private_keys()
    config_nid = {
        "id": "nid_default",
        "keys": {"pp": pp, "sk": sk_nid},
    }
    crypto_nid = SIFEDamgard(config_nid)
    ct_nid = crypto_nid.encrypt(x)
    assert isinstance(ct_nid, dict)
    assert "c" in ct_nid
    assert "d" in ct_nid
    assert "e" in ct_nid

    config_sid = {
        "id": "sid_0",
        "keys": {"pp": pp},
    }
    crypto_sid = SIFEDamgard(config_sid)
    dk = kg.get_decryption_keys(config_sid["id"], credentials={"fusion_weight": y})
    computed_inner_prod = crypto_sid.decrypt(ct_nid, dk, y)
    logger.info("<{},{}> expected inner-prod = {}".format(x, y, expected_inner_prod))
    logger.info("<{},{}> computed inner-prod = {}".format(x, y, computed_inner_prod))
    assert computed_inner_prod == expected_inner_prod


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound"] = 50

    kg = SIFEDamgardKeyGenerator(cfg)
    kg.setup()

    precision = 1
    fusion_weight = [3]
    lst_x = [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
    expected = [
        ((ary * pow(10, precision)).astype(int) / pow(10, precision))
        * fusion_weight[0]
        for ary in lst_x
    ]

    pp = kg.get_public_parameters()
    crypto_nid = SIFEDamgard(
        {
            "id": "nid_default",
            "precision": precision,
            "keys": {"pp": pp, "sk": kg.get_private_keys()},
        }
    )
    crypto_sid = SIFEDamgard(
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
