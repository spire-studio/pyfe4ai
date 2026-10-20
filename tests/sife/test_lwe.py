import logging
import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.sife.lwe import SIFELWE
from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEValidationError

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 4,
        "lwe_n": 16,
        "bound_x": 8,
        "bound_y": 8,
    }


def test_key_generator(kg_config):
    kg = SIFELWEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.eta == kg_config["eta"]
    assert kg.lwe_n == kg_config["lwe_n"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_SIFE_LWE)
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.p) == pp["p"]
    assert gp.digits(kg.q) == pp["q"]
    # A is published as a seed, not as its m x lwe_n entries
    assert "A" not in pp
    assert len(bytes.fromhex(pp["A_seed"])) == 32
    assert (pp["m"], pp["lwe_n"]) == (kg.m, kg.lwe_n)

    enc_key = kg.get_private_keys()
    assert "pk" in enc_key

    fusion_weight = [1 for _ in range(kg_config["eta"])]
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "sk_y" in dk
    assert len(dk["sk_y"]) == kg.lwe_n


@pytest.mark.parametrize("mode", ["memory", "stream"])
def test_entire_process(kg_config, mode):
    kg = SIFELWEKeyGenerator({**kg_config, "matrix_mode": mode})
    kg.setup()

    x = [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(4)]
    y = [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(4)]
    expected_inner_prod = sum(a * b for a, b in zip(x, y))

    pp = kg.get_public_parameters()
    sk_nid = kg.get_private_keys()
    crypto_nid = SIFELWE({"id": "nid_default", "keys": {"pp": pp, "sk": sk_nid}, "matrix_mode": mode})
    ct_nid = crypto_nid.encrypt(x)
    assert isinstance(ct_nid, dict)
    assert "ct0" in ct_nid
    assert "ct1" in ct_nid

    crypto_sid = SIFELWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    computed_inner_prod = crypto_sid.decrypt(ct_nid, dk, y)
    logger.info("<{},{}> expected inner-prod = {}".format(x, y, expected_inner_prod))
    logger.info("<{},{}> computed inner-prod = {}".format(x, y, computed_inner_prod))
    assert computed_inner_prod == expected_inner_prod


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = SIFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    fusion_weight = [2]
    lst_x = [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
    expected = [
        (np.rint(ary * pow(10, precision)) / pow(10, precision))
        * fusion_weight[0]
        for ary in lst_x
    ]

    pp = kg.get_public_parameters()
    crypto_nid = SIFELWE(
        {
            "id": "nid_default",
            "precision": precision,
            "keys": {"pp": pp, "sk": kg.get_private_keys()},
        }
    )
    crypto_sid = SIFELWE(
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


def test_unknown_matrix_mode_is_refused(kg_config):
    kg = SIFELWEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(FEValidationError, match="matrix_mode"):
        SIFELWE({"keys": {"pp": kg.get_public_parameters()}, "matrix_mode": "disk"})
