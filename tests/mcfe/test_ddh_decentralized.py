import os
import random
import pytest
import logging

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFE
from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.logger import init_logging_config

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    sec_param = 128
    n = 3
    s = 1
    _config = {
        "sec_param": sec_param,
        "lst_nid": ["nid_{}".format(i) for i in range(n)],
        "eta": 1,
        "n": n,
        "s": s,
    }
    return _config


def test_key_generator(kg_config):
    kg = DecentralizedMCFEKeyGenerator(kg_config)

    assert kg.sec_param == kg_config["sec_param"]
    assert kg.n == kg_config["n"]
    assert kg.s == kg_config["s"]
    assert kg.eta == kg_config["eta"]
    assert kg.lst_nid == kg_config["lst_nid"]

    config_folder = os.path.join("config", "authority", CryptoCONST.TYPE_DMCFE)
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
        assert "s" in sk_nid and isinstance(sk_nid["s"], list)
        assert "u" in sk_nid and isinstance(sk_nid["u"], list)
        assert "v" in sk_nid and isinstance(sk_nid["v"], list)


def test_entire_process(kg_config):
    lst_nid = kg_config["lst_nid"]

    precision = 3
    label = "test-label"

    kg = DecentralizedMCFEKeyGenerator(kg_config)
    kg.setup()

    # prepare input
    dct_x, dct_y = {}, {}
    for nid in lst_nid:
        sign = 1 if random.random() < 0.5 else -1
        dct_x[nid] = [sign * random.randint(1, 10)]
        dct_y[nid] = [random.randint(1, 10)]
    lst_x = list(dct_x.values())
    lst_y = list(dct_y.values())
    expected_inner_prod = sum(lst_x[i][0] * lst_y[i][0] for i in range(len(lst_x)))

    # encryption
    dct_ct = {}
    dct_dk_shares = {}
    _pp = kg.get_public_parameters()
    for nid in lst_nid:
        _sk_nid = kg.get_private_keys(nid)
        _config_nid = {
            "id": nid,
            "precision": precision,
            "keys": {"pp": _pp, "sk": _sk_nid},
        }
        crypto_nid = DecentralizedMCFE(_config_nid)
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid], label)
        dct_dk_shares[nid] = crypto_nid.derive_function_decryption_key_share(dct_y)
        assert isinstance(dct_ct[nid], dict)
        assert "ct0" in dct_ct[nid]
        assert "ct1" in dct_ct[nid]
        assert isinstance(dct_dk_shares[nid], dict)
        assert "dk0" in dct_dk_shares[nid]
        assert "dk1" in dct_dk_shares[nid]

    # decryption
    _config_sid = {
        "id": "sid_0",
        "precision": precision,
        "keys": {"pp": _pp},
    }
    crypto_sid = DecentralizedMCFE(_config_sid)
    dk = crypto_sid.combine_function_decryption_key_share(dct_dk_shares)
    assert isinstance(dk, dict)
    assert "dk0" in dk
    assert "dk1" in dk

    computed_inner_prod = crypto_sid.decrypt(dct_ct, dk, dct_y, label)
    logger.info(
        "<{},{}> expected inner-prod = {}".format(lst_x, lst_y, expected_inner_prod)
    )
    logger.info(
        "<{},{}> computed inner-prod = {}".format(lst_x, lst_y, computed_inner_prod)
    )
    assert computed_inner_prod == expected_inner_prod


def test_get_decryption_keys_raises_not_implemented(kg_config):
    """Regression (N12): the method used to *return* NotImplementedError."""
    kg = DecentralizedMCFEKeyGenerator(kg_config)
    kg.setup()
    with pytest.raises(NotImplementedError, match="derive_function_decryption_key_share"):
        kg.get_decryption_keys("sid_0", credentials={"fusion_weight": {}})
