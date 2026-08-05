import os
import time
import random
import pytest
import logging
from contextlib import contextmanager

import gmpy2 as gp

from pyfe4ai.schemes.sife.ddh import SIFE
from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST

from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@contextmanager
def timer(ctx_msg):
    """Helper for measuring runtime"""
    time0 = time.perf_counter()
    yield
    logger.info("[%s][elapsed time: %.2f s]" % (ctx_msg, time.perf_counter() - time0))


@pytest.fixture(params=[3, 5, 8], ids=["eta=3", "eta=5", "eta=8"])
def kg_config(request):
    return {
        "sec_param": 128,
        "eta": request.param,
    }


def test_key_generator(kg_config):
    kg = SIFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE)
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.p) == pp["p"]
    assert gp.digits(kg.g) == pp["g"]

    sk_nid = kg.get_private_keys()
    assert "gs" in sk_nid

    fusion_weight = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "z" in dk
    assert isinstance(dk["z"], str)


def test_entire_process(kg_config):
    kg = SIFEKeyGenerator(kg_config)
    kg.setup()
    eta = kg_config["eta"]
    precision = 3

    max_test_value = 100
    x = [random.randint(0, max_test_value) for i in range(eta)]
    y = [random.randint(0, max_test_value) for i in range(eta)]

    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    expected_inner_prod = sum(map(lambda i: x[i] * y[i], range(eta)))
    logger.debug("original dot product <x,y>: %d" % expected_inner_prod)

    # encryption
    pp = kg.get_public_parameters()
    sk_nid = kg.get_private_keys()
    config_nid = {
        "id": "nid_default",
        "precision": precision,
        "keys": {"pp": pp, "sk": sk_nid},
    }
    crypto_nid = SIFE(config_nid)
    ct_nid = crypto_nid.encrypt(x)
    assert isinstance(ct_nid, dict)
    assert "ct0" in ct_nid
    assert "ct1" in ct_nid

    # decryption
    config_sid = {
        "id": "sid_0",
        "precision": precision,
        "keys": {"pp": pp},
    }
    crypto_sid = SIFE(config_sid)
    dk = kg.get_decryption_keys(config_sid["id"], credentials={"fusion_weight": y})
    computed_inner_prod = crypto_sid.decrypt(ct_nid, dk, y)
    logger.info("<{},{}> expected inner-prod = {}".format(x, y, expected_inner_prod))
    logger.info("<{},{}> computed inner-prod = {}".format(x, y, computed_inner_prod))
    assert computed_inner_prod == expected_inner_prod
