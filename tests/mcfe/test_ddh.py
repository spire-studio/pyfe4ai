import os
import random
import pytest
import logging

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh import MCFE
from pyfe4ai.schemes.mcfe.ddh import MCFEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture(params=[(3, 1), (2, 2)], ids=["n=3,eta=1", "n=2,eta=2"])
def kg_config(request):
    n, eta = request.param
    return {
        "sec_param": 256,
        "lst_nid": ["nid_{}".format(i) for i in range(n)],
        "eta": eta,
        "n": n,
        "s": 1,
    }


def test_key_generator(kg_config):
    kg = MCFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.eta == kg_config["eta"]
    assert kg.lst_nid == kg_config["lst_nid"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_MCFE)
    assert os.path.exists(config_folder)
    param_file = os.path.join(config_folder, "param.json")
    assert os.path.exists(param_file)

    lst_nid_expected = ["nid_{}".format(i) for i in range(kg_config["n"])]
    assert kg.lst_nid == lst_nid_expected

    kg.setup()
    pp = kg.get_public_parameters()
    assert gp.digits(kg.p) == pp["p"]
    assert gp.digits(kg.g) == pp["g"]

    for nid in lst_nid_expected:
        sk_nid = kg.get_private_keys(nid)
        assert "w" in sk_nid
        assert "u" in sk_nid

    fusion_weight = {
        nid: [1 for _ in range(kg_config["eta"])] for nid in lst_nid_expected
    }
    dk = kg.get_decryption_keys(None, credentials={"fusion_weight": fusion_weight})
    assert "d" in dk
    assert "z" in dk
    assert isinstance(dk["d"], dict)
    assert isinstance(dk["z"], str)


def test_entire_process(kg_config):
    lst_nid = kg_config["lst_nid"]

    precision = 3
    label = "test-label"

    kg = MCFEKeyGenerator(kg_config)
    kg.setup()

    # prepare input
    dct_x, dct_y = {}, {}
    for nid in lst_nid:
        sign = 1 if random.random() < 0.5 else -1
        dct_x[nid] = [sign * random.randint(1, 10)]
        dct_y[nid] = [1]
    lst_x = list(dct_x.values())
    lst_y = list(dct_y.values())
    expected_inner_prod = sum(lst_x[i][0] * lst_y[i][0] for i in range(len(lst_x)))

    # encryption
    dct_ct = {}
    _pp = kg.get_public_parameters()
    for nid in lst_nid:
        _sk_nid = kg.get_private_keys(nid)
        _config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": _pp, "sk": _sk_nid},
        }
        crypto_nid = MCFE(_config_nid)
        ct_nid = crypto_nid.encrypt(dct_x[nid], label)
        assert isinstance(ct_nid, dict)
        assert "t" in ct_nid
        assert "c" in ct_nid
        dct_ct[nid] = ct_nid

    # decryption
    _config_sid = {
        "id": "sid_0",
        "precision": precision,
        "keys": {"pp": _pp},
    }
    crypto_sid = MCFE(_config_sid)
    dk = kg.get_decryption_keys(_config_sid["id"], credentials={"fusion_weight": dct_y})
    computed_inner_prod = crypto_sid.decrypt(dct_ct, dk, dct_y, label)
    logger.info(
        "<{},{}> expected inner-prod = {}".format(lst_x, lst_y, expected_inner_prod)
    )
    logger.info(
        "<{},{}> computed inner-prod = {}".format(lst_x, lst_y, computed_inner_prod)
    )
    assert computed_inner_prod == expected_inner_prod
